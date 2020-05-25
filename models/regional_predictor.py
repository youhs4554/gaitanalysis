import torch
from torch import nn
import torch.nn.functional as F
import random

__all__ = [
    'RegionalPredictor'
]


class RegionalPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task='classification'):
        if task != 'classification':
            raise ValueError(
                'Invalid task type, this class works only for classification. You entered {}'.format(task))

        super(RegionalPredictor, self).__init__()
        self.m = nn.Conv2d(n_inputs, n_outputs, 1)
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *inputs):

        x, targets, masks = inputs
        x = x.mean(2)  # temporal average pooling

        out = self.m(x)
        if targets is None:
            # detach background cls
            return F.max_pool2d(out[:, 1:], kernel_size=out.size(-1)).flatten(1), 0.0

        masks = F.interpolate(masks, size=(1,)+x.size()
                              [-2:], mode='nearest').squeeze(2)  # (N,1,7,7)
        targets_grid = torch.zeros_like(masks).repeat(
            (1, self.n_outputs, 1, 1))  # (N,nclass+1,7,7)

        targets = targets.view(
            [-1] + [1, ]*(targets_grid.dim()-1))  # (N,1,1,1)

        # blend targets & masks
        blended_targets = masks * (targets+1)  # for b.g class space
        target_grid = targets_grid.scatter(1, blended_targets.long(), 1.0)

        loss_dict = {
            self.task: self.criterion(out, target_grid.argmax(1))
        }

        # detach background cls
        return F.max_pool2d(out[:, 1:], kernel_size=out.size(-1)).flatten(1), loss_dict
