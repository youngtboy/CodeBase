import os
import argparse
import torch
from torch.utils.data import DataLoader
import time
import torch.optim as optim
import torch.nn as nn
import tempfile
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torch.distributed as dist

from src.datasets import *
from utils.train_utils import train_one_epoch
from utils.validate_utils import validate
from utils.misc import get_logger, collect_env, init_distributed_mode, init_random_seed, set_random_seed
from utils.lr_updater import LrUpdater

from src.models.segmentors import EXP1Model as Model
from src.datasets.samplers import GroupSampler

from src.losses import binary_cross_entropy, structure_loss
from src.datasets import samplers
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')

    parser.add_argument('--syncbn', default=False, type=bool)
    # TODO add ema-related codes, the mode is now unavailable
    parser.add_argument('--ema', default=False, type=bool)
    # TODO add amp-related codes, the mode is now unavailable
    parser.add_argument('--amp', default=False, type=bool)

    # seed config
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff-seed', action='store_true', help='Whether or not set different seeds for different ranks')

    # path config
    parser.add_argument('--data-root', default='./data/NJUD_NLPR_DUT')
    parser.add_argument('--img-dir', default='Image')
    parser.add_argument('--depth-dir', default='Depth')
    parser.add_argument('--ann_dir', default='GT')

    parser.add_argument('--work-dir', default="./work_dirs/dist_drop", help='the dir to save logs and models')

    # device config
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local_rank', type=int, default=0)
    

    # train & val config
    parser.add_argument('--epochs', default=100)

    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--nw', default=8)

    parser.add_argument('--interval', default=5, help='the interval for validation and saving checkpoint')

    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')

    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')

    # lr config
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--lr_policy', default='poly')

    # optimizer
    parser.add_argument('--wd', default=0.0005)
    parser.add_argument('--momentum', default=0.9)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed_mode(args)

    seed = init_random_seed(args.seed, device=args.device)
    seed = seed + args.local_rank if args.diff_seed else seed
    set_random_seed(seed)
    
    if args.local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)

    assert torch.cuda.is_available()
    device = torch.device(args.device)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger = get_logger(log_file=log_file)

    logger.info(f"Launch distribute mode: {args.distributed}")

    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    config_info = '\n'.join([(f'{k}: {v}') for k, v, in vars(args).items() if k not in ["local_rank", "gpu", "rank"]])
    logger.info('Config info:\n' + config_info + '\n' + dash_line)

    tb_writer = SummaryWriter(log_dir=os.path.join(args.work_dir, "tf_log"), filename_suffix='_' + timestamp)
    logger.info(f'TensorBoard infos are saved in {tb_writer.log_dir}')

    data_transform = {
        'train': [LoadImageFromFile(), LoadAnnotations(), LoadDepthFromFile(color_type=1),
                  Resize(img_scale=(352, 352)), RandomFlip(prob=0.5),
                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  ImageToTensor(['img', 'depth_map', 'gt_semantic_seg']),
                  Collect(['img', 'depth_map', 'gt_semantic_seg'])],
        'val': [LoadImageFromFile(), LoadAnnotations(), LoadDepthFromFile(color_type=1),
                Resize(img_scale=(352, 352)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ImageToTensor(['img', 'depth_map', 'gt_semantic_seg']),
                Collect(['img', 'depth_map', 'gt_semantic_seg'])]
    }

    train_dt_info = '\n'.join([(f'{dt}') for dt in data_transform['train']])
    logger.info('Train Data Transforms info:\n' + train_dt_info + '\n' + dash_line)
    val_dt_info = '\n'.join([(f'{dt}') for dt in data_transform['val']])
    logger.info('Val Data Transforms info:\n' + val_dt_info + '\n' + dash_line)

    dataset_train = SODDataset(pipeline=data_transform['train'],
                           data_root=args.data_root,
                           img_dir=args.img_dir,
                           ann_dir=args.ann_dir,
                           depth_dir=args.depth_dir)
    logger.info(f'Loading {len(dataset_train)} images for training')

    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = samplers.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    if sampler_train is not None:
        logger.info(f"Using {sampler_train.__class__.__name__} as sampler for Train Dataloader")

    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   num_workers=args.nw,
                                   collate_fn=dataset_train.collate_fn,
                                   pin_memory=True)

    val_dataset_NJUD = SODDataset(pipeline=data_transform['val'],
                                  data_root="./data/Test_Data/NJUD",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)
    val_dataset_NLPR = SODDataset(pipeline=data_transform['val'],
                                  data_root="./data/Test_Data/NLPR",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)
    val_dataset_SSD = SODDataset(pipeline=data_transform['val'],
                                 data_root="./data/Test_Data/SSD",
                                 img_dir=args.img_dir,
                                 ann_dir=args.ann_dir,
                                 depth_dir=args.depth_dir)
    val_dataset_SIP = SODDataset(pipeline=data_transform['val'],
                                 data_root="./data/Test_Data/SIP",
                                 img_dir=args.img_dir,
                                 ann_dir=args.ann_dir,
                                 depth_dir=args.depth_dir)
    val_dataset_STERE = SODDataset(pipeline=data_transform['val'],
                                   data_root="./data/Test_Data/STERE",
                                   img_dir=args.img_dir,
                                   ann_dir=args.ann_dir,
                                   depth_dir=args.depth_dir)
    val_dataset_DES = SODDataset(pipeline=data_transform['val'],
                                 data_root="./data/Test_Data/DES",
                                 img_dir=args.img_dir,
                                 ann_dir=args.ann_dir,
                                 depth_dir=args.depth_dir)
    val_dataset_DUTD = SODDataset(pipeline=data_transform['val'],
                                  data_root="./data/Test_Data/DUTD",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)
    val_dataset_LFSD = SODDataset(pipeline=data_transform['val'],
                                  data_root="./data/Test_Data/LFSD",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)

    val_datasets = [val_dataset_NLPR, val_dataset_NJUD, val_dataset_SSD, val_dataset_SIP, val_dataset_STERE,
                   val_dataset_DES, val_dataset_DUTD, val_dataset_LFSD]

    val_dataloaders = []
    for val_dataset in val_datasets:
        val_dataloaders.append(DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=args.nw,
                                          collate_fn=val_dataset.collate_fn,
                                          pin_memory=True))

    model = Model().to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=args.wd)
    logger.info(f'Using {optimizer.__class__.__name__} as optimizer')

    lrupdater = LrUpdater(args.lr_policy, optimizer=optimizer)
    logger.info(f'Using {lrupdater.mode} policy to adjust lr')


    start_epoch = 0

    if args.distributed:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pth")
        if args.local_rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        torch.distributed.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.load_from is not None:
        assert os.path.exists(args.load_from), "Error: The checkpoint does not exist"
        logger.info(f"load from {args.load_from} as pretrain model")
        loaded_dict = torch.load(args.load_from, map_location=device)
        model.load_state_dict(loaded_dict)

    if args.resume_from is not None:
        assert os.path.exists(args.resume_from), "Error: The checkpoint does not exist"
        logger.info(f"resume from {args.resume_from}")
        loaded_dict = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(loaded_dict["model"])
        optimizer.load_state_dict(loaded_dict["optimizer"])
        start_epoch = loaded_dict["epoch"] + 1

    if args.distributed:
        if args.syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.gpu])

    # criterion = binary_cross_entropy
    criterion = structure_loss
    logger.info(f'Using {criterion.__name__} as loss function')


    logger.info(f'Start Training Workflow Running, max {args.epochs} epochs')
    logger.info(f'Checkpoints will be saved to {args.work_dir}')
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(args,
                        model=model,
                        optimizer=optimizer,
                        lrupdater=lrupdater,
                        criterion=criterion,
                        train_loader=data_loader_train,
                        device=device,
                        epoch=epoch,
                        logger=logger,
                        tb=tb_writer)

        if ((epoch+1) % args.interval) == 0:

            infos = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            # save weights
            if args.local_rank == 0:
                torch.save(infos, os.path.join(args.work_dir, f"Epoch_{epoch}" + ".pth"))
                logger.info(f"Checkpoint Epoch_{epoch}.pth has been saved")

            if not args.no_validate:
                logger.info('Start Validation Workflow Running')
                timestart = time.time()
                validate(model, val_dataloaders, args.work_dir, logger, device, tb_writer)
                timeend = time.time()
                if args.distributed:
                    torch.distributed.barrier()
                logger.info(f'Validation Workflow has been finished, {round((timeend - timestart)/60, 2)} min')

    if args.distributed:
        torch.distributed.destroy_process_group()
    logger.info('Training Workflow has been finished')


if __name__ == '__main__':
    main()