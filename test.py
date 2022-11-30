import argparse
from src.datasets import *
import torch
from torch.utils.data import DataLoader
from src.models.segmentors import EXP1Model as Model
import os
from utils.validate_utils import validate


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--data-root', default='data/NJUD_NLPR_DUT')
    parser.add_argument('--img-dir', default='Image')
    parser.add_argument('--depth-dir', default='Depth')
    parser.add_argument('--ann_dir', default='GT')

    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--nw', default=8)

    parser.add_argument('--work-dir', default="./workdir/exp1", help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    data_transform = [LoadImageFromFile(), LoadAnnotations(), LoadDepthFromFile(color_type=1),
                      Resize((352, 352)),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ImageToTensor(['img', 'depth_map', 'gt_semantic_seg']),
                      Collect(['img', 'depth_map', 'gt_semantic_seg'])]


    val_dataset_NJUD = SODDataset(pipeline=data_transform,
                                  data_root="./data/Test_Data/NJUD",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)
    val_dataset_NLPR = SODDataset(pipeline=data_transform,
                                  data_root="./data/Test_Data/NLPR",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)
    val_dataset_SSD = SODDataset(pipeline=data_transform,
                                 data_root="./data/Test_Data/SSD",
                                 img_dir=args.img_dir,
                                 ann_dir=args.ann_dir,
                                 depth_dir=args.depth_dir)
    val_dataset_SIP = SODDataset(pipeline=data_transform,
                                 data_root="./data/Test_Data/SIP",
                                 img_dir=args.img_dir,
                                 ann_dir=args.ann_dir,
                                 depth_dir=args.depth_dir)
    val_dataset_STERE = SODDataset(pipeline=data_transform,
                                   data_root="./data/Test_Data/STERE",
                                   img_dir=args.img_dir,
                                   ann_dir=args.ann_dir,
                                   depth_dir=args.depth_dir)
    val_dataset_DES = SODDataset(pipeline=data_transform,
                                 data_root="./data/Test_Data/DES",
                                 img_dir=args.img_dir,
                                 ann_dir=args.ann_dir,
                                 depth_dir=args.depth_dir)
    val_dataset_DUTD = SODDataset(pipeline=data_transform,
                                  data_root="./data/Test_Data/DUTD",
                                  img_dir=args.img_dir,
                                  ann_dir=args.ann_dir,
                                  depth_dir=args.depth_dir)
    val_dataset_LFSD = SODDataset(pipeline=data_transform,
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
                                          num_workers=args.nw,
                                          collate_fn=val_dataset.collate_fn))

    model = Model().cuda()

    if args.load_from is not None:
        assert os.path.exists(args.load_from), "Error: The checkpoint does not exist"
        print("load from {} to test".format(args.load_from))
        loaded_dict = torch.load(args.load_from, map_location=device)
        model.load_state_dict(loaded_dict["model"])

    validate(model, val_dataloaders, args.work_dir)


if __name__ == '__main__':
    main()