import torch
from prettytable import PrettyTable
from tqdm import tqdm
import os
import torch.nn.functional as F
import cv2
import time
from utils.misc import get_rank


def validate(model, val_loaders, workdir, logger, device, tb_writer):
    rank = get_rank()
    model.eval()
    with torch.no_grad():
        for val_loader in val_loaders:
            dataset_name = val_loader.dataset.img_dir.split("Image")[0].split("/")[-2]
            if rank == 0:
                val_loader = tqdm(val_loader, desc=dataset_name+" is validating")
            save_dir = os.path.join(workdir, "out", dataset_name)
            for _, data in enumerate(val_loader):
                img, depth, _, img_meta = data
                img, depth = img.to(device), depth.float().to(device)
                size = img_meta[0]["ori_shape"][:-1]
                pred = model(img, depth)
                pred = F.interpolate(
                    input=pred, size=size,
                    mode='bilinear', align_corners=False
                )

                img_name = img_meta[0]["ori_filename"].replace("jpg", "png")
                os.makedirs(save_dir, exist_ok=True)

                res = pred.sigmoid().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(os.path.join(save_dir, str(img_name)), res * 255)

            _result = getMetrics("./data/Test_Data/{}/GT".format(dataset_name),  # ground truth
                                 save_dir)  # prediction
            _table = PrettyTable(["Sm", "maxFm", "meanFm",  "adpFm", "wFm", "maxEm", "meanEm", "adpEm", "MAE"])
            _table.title = dataset_name
            _table.add_row([round(_result["Smeasure"], 3),
                            round(_result["maxFm"], 3),
                            round(_result["meanFm"], 3),
                            round(_result["adpFm"], 3),
                            round(_result["wFmeasure"], 3),
                            round(_result["maxEm"], 3),
                            round(_result["meanEm"], 3),
                            round(_result["adpEm"], 3),
                            round(_result["MAE"], 3)])

            logger.info(dataset_name + ' dataset validation results' + '\n' + _table.get_string())


def getMetrics(mask_root, pred_root):
    from utils.metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE = MAE()

    mask_root = os.path.join(mask_root)
    pred_root = os.path.join(pred_root)
    pred_name_list = sorted(os.listdir(pred_root))
    if get_rank() == 0:
        pred_name_list = tqdm(pred_name_list)
    for mask_name in pred_name_list:
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
    }
    return results