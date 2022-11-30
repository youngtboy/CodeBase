import torch.nn.functional as F
import torch


def binary_cross_entropy(pred,
                         label,
                         class_weight=None,):

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='mean')

    return loss



# copy from https://github.com/weijun88/F3Net/blob/eecace3adf1e8946b571a4f4397681252f9dc1b8/src/train.py#L18
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()