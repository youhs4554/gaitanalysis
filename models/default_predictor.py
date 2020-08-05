import torch
from torch import nn
import random
import math
from .losses import FocalLoss, LabelSmoothLoss

__all__ = ["DefaultPredictor"]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task="regression"):
        super(DefaultPredictor, self).__init__()
        self.classifier = nn.Sequential(
                                        nn.Dropout(0.5), 
                                        nn.Linear(n_inputs, n_outputs)
                                        )

        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if task == "regression":
            self.criterion = nn.SmoothL1Loss()
        elif task == "classification":
            self.criterion = LabelSmoothLoss(smoothing=0.1)

    def forward(self, *args, **kwargs):
        x, targets, averaged, lambda_ = args

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.classifier(x)
        if targets is None:
            return out, 0.0

        if averaged:
            bs = targets.size(0)
            nclips = x.size(0) // bs

            out = out.view(bs, nclips, -1).mean(1)  # avg over crops

        if isinstance(targets, list):
            # mixup
            target_a, target_b = targets
            loss_val = lambda_ * self.criterion(out, target_a) + (1-lambda_) * self.criterion(out, target_b)
        else:
            loss_val = self.criterion(out, targets)

        loss_dict = {self.task: loss_val}

        return out, loss_dict
