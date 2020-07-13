import torch
from torch import nn
import random
import math
from .losses import FocalLoss

__all__ = ["DefaultPredictor"]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task="regression"):
        super(DefaultPredictor, self).__init__()
        self.m = nn.Sequential(nn.Dropout(0.5), nn.Linear(n_inputs, n_outputs))
        # self.merge = nn.Linear(2, 1)
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if task == "regression":
            self.criterion = nn.SmoothL1Loss()
        elif task == "classification":
            self.criterion = FocalLoss()  # use focal loss for class balancing...

    def forward(self, *args, **kwargs):
        x, targets, averaged = args

        bs = targets.size(0)
        nclips = x.size(0) // bs

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.m(x)
        if targets is None:
            return out, 0.0

        if averaged:
            out = out.view(bs, nclips, -1).mean(1)  # avg over crops

        if kwargs.get("appearance_pred") is not None:
            # fusion with predictions from `appearance`!
            # TODO. use mlp? to determine combinationweights?
            # out = torch.stack((out, kwargs.get("appearance_pred")), -1)
            # out = self.merge(out).squeeze(-1)
            pass
        loss_dict = {self.task: self.criterion(out, targets)}

        return out, loss_dict
