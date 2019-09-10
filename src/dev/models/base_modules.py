import torch
from torch import nn
from torch.nn.functional import max_pool2d


class Concat(nn.Module):
    def __init__(self, layers, dim=-1):
        super(Concat, self).__init__()
        self.layers = layers
        self.dim = dim

    def forward(self, x):
        res = []
        for l in self.layers:
            res.append(l(x))

        return torch.cat(res, dim=self.dim)


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


class GMP(nn.Module):
    def __init__(self, keepdim=True):
        super(GMP, self).__init__()
        self.keepdim = keepdim

    def forward(self, x):
        x = max_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        if not self.keepdim:
            x = max_pool2d(x, kernel_size=(x.size(2), x.size(3))).view(
                x.size(0), -1)  # (b,C))
        return x


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:  # tensor case
                inputs = module(inputs)
        return inputs
