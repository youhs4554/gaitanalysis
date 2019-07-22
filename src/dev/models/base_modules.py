import torch
from torch import nn

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class GAP(nn.Module):
    def __init__(self, *dims, keepdim=True):
        super(GAP, self).__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, self.dims, keepdim=self.keepdim)