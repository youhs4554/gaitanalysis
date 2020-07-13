import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.metrics import f1_score

##### Calssification losses #######


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


class F1_Loss(nn.Module):
    def __init__(self, eps=1e-7):
        super(F1_Loss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        preds = preds.argmax(1)
        TP = torch.sum((preds == 1) & (targets == 1)).float()
        TN = torch.sum((preds == 0) & (targets == 0)).float()
        FP = torch.sum((preds == 1) & (targets == 0)).float()
        FN = torch.sum((preds == 0) & (targets == 1)).float()

        # binary case, i.e. only consider postive class
        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)

        F1 = 2 * (precision * recall) / (precision + recall + self.eps)

        return 1 - F1


#### Mask prediction losses ####
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class MultiScaled_BCELoss(nn.Module):
    def __init__(self, n_scales):
        super(MultiScaled_BCELoss, self).__init__()
        self.n_scales = n_scales

    def forward(self, predict, target):
        l = []
        for i in range(self.n_scales):
            shapes = predict[i].size()[2:]
            target = F.interpolate(target, size=shapes)
            l.append(F.binary_cross_entropy(predict[i], target))

        return torch.stack(l).mean()


class MultiScaled_DiceLoss(nn.Module):
    def __init__(self, n_scales):
        super(MultiScaled_DiceLoss, self).__init__()
        self.n_scales = n_scales
        self.dice = BinaryDiceLoss()

    def forward(self, predict, target):
        l = []
        for i in range(self.n_scales):
            shapes = predict[i].size()[2:]
            target = F.interpolate(target, size=shapes, mode="area")
            _dice = self.dice(predict[i].argmax(1).float(), target.argmax(1).float())
            l.append(_dice)

        return torch.stack(l).mean()
