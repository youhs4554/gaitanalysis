import torch
from torch import nn
import random
import math
from .losses import FocalLoss, LabelSmoothLoss

__all__ = ["DefaultPredictor"]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task="regression"):
        super(DefaultPredictor, self).__init__()
        self.classifier = nn.Linear(n_inputs, n_outputs)
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if task == "regression":
            self.criterion = nn.SmoothL1Loss()
        elif task == "classification":
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        x, targets, lambda_ = args

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.classifier(x)
        if targets is None:
            return out, 0.0

        if isinstance(targets, list):
            # mixup
            target_a, target_b = targets
            loss_val = lambda_ * self.criterion(out, target_a) + (
                1 - lambda_
            ) * self.criterion(out, target_b)
        else:
            loss_val = self.criterion(out, targets)

        loss_dict = {self.task: loss_val}

        return out, loss_dict
