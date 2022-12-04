import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
from utils.misc import reduce_tensor, get_rank 
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(args,
                    model,
                    optimizer,
                    lrupdater,
                    criterion,
                    train_loader,
                    device,
                    epoch,
                    logger,
                    tb
                    ):
    model.train()
    rank = get_rank()
    mean_epoch_loss = torch.zeros(1).to(device)
    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    log_info = {"Epoch": epoch}
    if rank == 0:
        train_loader = tqdm(train_loader)

    timestart = time.time()
    for _inner_iter, data in enumerate(train_loader):
        _iter = _inner_iter + epoch * iters_per_epoch

        img, depth, gt, _ = data
        img, depth, gt = img.to(device), depth.float().to(device), gt.to(device)

        if _inner_iter == 0:
            lr_start_epoch = optimizer.param_groups[0]['lr']
            log_info["LR"] = '%.3e' % lr_start_epoch
        lr_updater = getattr(lrupdater, f'adjust_lr_{lrupdater.mode}')
        lr_updater(optimizer, _iter, total_iters)

        if args.local_rank == 0:
            tb.add_scalar("LR", optimizer.param_groups[0]["lr"], _iter)

        if args.amp:
            grad_scaler = GradScaler()

            with autocast(dtype=torch.float16):

                out = model(img, depth)
                out = F.interpolate(out, size=gt.size()[-2:], mode="bilinear", align_corners=False)
                loss = criterion(out, gt)
                reduce_tensor(loss, average=True)

            optimizer.zero_grad()
            
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        else:
            out = model(img, depth)
            out = F.interpolate(out, size=gt.size()[-2:], mode="bilinear", align_corners=False)

            loss = criterion(out, gt)

            reduce_tensor(loss, average=True)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        mean_epoch_loss = (mean_epoch_loss * _inner_iter + loss.detach()) / (_inner_iter + 1)

        if rank == 0:
            train_loader.desc = f"[Epoch {epoch}] mean loss {round(mean_epoch_loss.item(), 4)}"

    timeend = time.time()
    
    if args.rank == 0:
        tb.add_scalar("Loss", mean_epoch_loss, epoch)
    log_info["Loss"] = round(mean_epoch_loss.item(), 4)
    log_info['Time'] = round((timeend - timestart)/60, 2)

    logger.info("Epoch [{}]  Time: {} min  Lr: {}  Loss: {}".format(log_info["Epoch"], log_info["Time"], log_info["LR"], log_info["Loss"]))