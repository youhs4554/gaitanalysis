import torch
from torch import nn
import random
import math
from .losses import FocalLoss

__all__ = ["DefaultPredictor"]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task="regression"):
        super(DefaultPredictor, self).__init__()
        self.classifier = nn.Sequential(
                                        nn.Dropout(0.5), 
                                        nn.Linear(n_inputs, n_outputs)
                                        )

        self.merge = nn.Linear(2, 1)
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if task == "regression":
            self.criterion = nn.SmoothL1Loss()
        elif task == "classification":
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        x, targets, averaged = args

        bs = targets.size(0)
        nclips = x.size(0) // bs

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.classifier(x)
        if targets is None:
            return out, 0.0

        if averaged:
            out = out.view(bs, nclips, -1).mean(1)  # avg over crops

        if kwargs.get("motion_pred") is not None:
            out = torch.stack((out, kwargs.get("motion_pred")), -1)
            out = self.merge(out).squeeze(-1)

        loss_dict = {self.task: self.criterion(out, targets)}

        return out, loss_dict
