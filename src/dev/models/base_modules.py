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


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, upscale_input=True):
        super().__init__()

        if upscale_input:
            if interpolate:
                self.up = nn.Upsample(
                    scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose3d(in_ch,
                                             in_ch//2, 2, stride=2, bias=False)

        self.upscale_input = upscale_input

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x1, x2):
        if self.upscale_input:
            x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x
