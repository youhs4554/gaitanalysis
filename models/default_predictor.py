import torch
from torch import nn
import random

__all__ = [
    'DefaultPredictor'
]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task='regression'):
        super(DefaultPredictor, self).__init__()
        self.m = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_inputs, n_outputs),
        )
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if task == 'regression':
            self.criterion = nn.SmoothL1Loss()
        elif task == 'classification':
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, *inputs):
        x, targets, _ = inputs

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.m(x)
        if targets is None:
            return out, 0.0

        loss_dict = {
            self.task: self.criterion(out, targets)
        }

        return out, loss_dict
