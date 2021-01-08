from models.losses import FocalLoss
import torch
from torch import nn
import random

__all__ = [
    'DefaultPredictor'
]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task='regression'):
        super(DefaultPredictor, self).__init__()
        self.m = nn.Linear(n_inputs, n_outputs)
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if task == 'regression':
            self.criterion = nn.SmoothL1Loss()
        elif task == 'classification':
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, *inputs):
        x, targets, enable_tsn, batch_size = inputs

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.m(x)

        if enable_tsn:
            # consensus
            out = out.view(
                batch_size, -1, out.size(-1)).mean(1)

        if targets is None:
            return out, 0.0

        loss_dict = {
            self.task: self.criterion(out, targets)
        }

        return out, loss_dict
