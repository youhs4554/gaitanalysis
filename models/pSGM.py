import torch
from torch import nn
from torch.nn import functional as F


class pSGM(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True,
    ):
        super(pSGM, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, mask_pred):
        """
        :param x: (b, c, t, h, w)
        :param mask_pred: (b, c, t, h, w)
        :return:
        """

        batch_size, dim = x.size()[:2]

        # g_x  : (b,inter,t,h,w) -> (b, inter, t*h*w) -> (b, t*h*w, inter)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta : (b,inter,t,h,w) -> (b, inter, t*h*w) -> (b, t*h*w, inter)
        theta = self.theta(mask_pred).view(batch_size, self.inter_channels, -1)
        theta = theta.permute(0, 2, 1)

        # phi : (b,inter,t,h,w) -> (b, inter, t*h*w)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta, phi)
        f_div_C = F.softmax(f, dim=1)  # (b, t*h*w, t*h*w)

        y = torch.matmul(f_div_C, g_x)  # (b, t*h*w, inter)
        y = y.permute(0, 2, 1).contiguous()  # (b, inter, t*h*w)
        y = y.view(
            batch_size, self.inter_channels, *x.size()[2:]
        )  # (b, inter, t, h, w)
        W_y = self.W(y)  # (b, c_in, t, h, w)
        z = W_y + x

        return z
