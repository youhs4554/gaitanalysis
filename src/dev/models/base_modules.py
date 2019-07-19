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


class Residual(nn.Module):
    def __init__(self, C_in, C_out):
        super(Residual, self).__init__()

        self.conv_1x1 = nn.Sequential(nn.Conv3d(C_in, C_out, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm3d(C_out),
                                      nn.ReLU(True)
                                      )

        self.model = nn.Sequential(
            nn.Conv3d(C_out, C_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(True),

            nn.Conv3d(C_out, C_out // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(C_out // 4),
            nn.ReLU(True),

            nn.Conv3d(C_out // 4, C_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(True),
        )

    def forward(self, x):
        res = self.conv_1x1(x)
        x = self.model(res)

        return res + x