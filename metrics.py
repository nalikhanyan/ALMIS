import torch.nn.functional as F
import torch


def calc_f1(pred, target, smooth=1):
    prd = torch.round(pred)
    intersection = (prd * target).sum()
    f1 = (2. * intersection) / (prd.sum() + target.sum() + smooth)
    f1_mean = f1.mean()
    return f1_mean


# def calc_IOU(pred, target, smooth=1e-7):
#     prd = torch.round(pred)
#     intersection = torch.logical_and(prd, targ).sum()
#     union = torch.logical_or(prd, targ).sum()
#     iou_score = (intersection + smooth) / (union + smooth)
#     iou_score_mean = iou_score.mean()
#     return iou_score_mean


def dice_loss(y_pred, y_true, smooth=1.):
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return 1. - dsc


def calc_loss(pred, target, bce_weight=1.):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    loss_bce = bce.data.cpu().numpy()
    loss_dice = dice.data.cpu().numpy()
    loss_sum = loss.data.cpu().numpy()

    return loss, loss_sum, loss_bce, loss_dice


def print_metrics(writer, metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
