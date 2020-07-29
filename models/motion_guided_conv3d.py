import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .losses import EpeLoss
from .i3res import inflated_resnet
from .default_predictor import DefaultPredictor
from collections import OrderedDict
import flow_vis
import numpy as np
import kornia

class ConvBlock(nn.Sequential):
    def __init__(self, conv_layer):
        super().__init__()
        self.add_module("conv", conv_layer)
        self.add_module("bn", nn.BatchNorm2d(conv_layer.out_channels))
        self.add_module("lReLU", nn.LeakyReLU(0.2, inplace=True))

class TinyMotionNet(nn.Module):
    def __init__(self, n_frames=16):
        super().__init__()
        in_channel = (n_frames + 1) * 3
        out_channel = n_frames * 2

        self.n_frames = n_frames

        self.conv1 = ConvBlock(nn.Conv2d(
            in_channel, 64, kernel_size=7, stride=1, padding=3, bias=False
        ))  # 112,112
        self.conv2 = ConvBlock(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False))  # 56, 56
        self.conv3 = ConvBlock(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False))  # 28, 28
        self.conv4 = ConvBlock(nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False))  # 14, 14

        self.flow4 = nn.Conv2d(
            128, out_channel, kernel_size=3, stride=1, padding=1
        )  # 14, 14
        self.deconv3 = ConvBlock(nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=False
        ))  # 28,28
        self.xconv3 = ConvBlock(nn.Conv2d(
            128 + out_channel + 256, 128, kernel_size=3, stride=1, padding=1, bias=False
        ))  # 28,28
        self.flow3 = nn.Conv2d(
            128, out_channel, kernel_size=3, stride=1, padding=1
        )  # 28, 28
        self.deconv2 = ConvBlock(nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=False
        ))  # 56, 56
        self.xconv2 = ConvBlock(nn.Conv2d(
            64 + out_channel + 128, 64, kernel_size=3, stride=1, padding=1, bias=False
        ))  # 56, 56
        self.flow2 = nn.Conv2d(
            64, out_channel, kernel_size=3, stride=1, padding=1
        )  # 56, 56

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

    def compute_loss(self, flow, label):

        # downsample label flows
        label = F.adaptive_avg_pool2d(label, flow.shape[2:])

        # minimizes reconstruction loss(=EpeLoss known as Pseudo-HuberLoss)
        flow_loss = EpeLoss()(pred=flow, label=label).mean()

        flow_x, flow_y = flow.split(
            self.n_frames, 1
        )  # flow_x(and y) : (B,n_frames,H,W)

        # each derivatives : (B, n_frames, 1, H, W)
        flow_x_dx, flow_x_dy = kornia.filters.SpatialGradient()(flow_x).split(1, 2)
        flow_y_dx, flow_y_dy = kornia.filters.SpatialGradient()(flow_y).split(1, 2)

        smooth_loss = torch.cat(
            [
                EpeLoss()(pred=x.squeeze(2), label=0).mean().unsqueeze(0)
                for x in [flow_x_dx, flow_x_dy, flow_y_dx, flow_y_dy]
            ],
            0,
        ).sum()

        B, nx2, H, W = flow.shape  # nx2 : each flow has 2-dimensions (x,y)
        flow_3d = flow.view(B, 2, -1, H, W)  # 2d -> 3d

        return flow_loss, smooth_loss, flow_3d

    def forward(self, x, label):
        # x : video frames / (N,C,n_frames+1,H,W)
        # label : optical-flow frames / (N,2,n_frames,H,W)

        # 3D -> 2D : stacks frames along channel dims
        B, _, _, H, W = x.shape
        x = x.view(B, -1, H, W)
        label = label.view(B, -1, H, W)

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        flow4 = torch.tanh(self.flow4(c4))  # flow4 : stacks of (14,14) w/ L=16
        flow_loss4, smooth_loss4, flow4_3d = self.compute_loss(flow4, label)

        # upsample flow4
        flow4_up = self.upsample(flow4)

        dc3 = self.deconv3(c4)
        xc3 = self.xconv3(torch.cat((dc3, flow4_up, c3), 1))

        flow3 = torch.tanh(self.flow3(xc3))  # flow3 : stacks of (28,28) w/ L=16
        flow_loss3, smooth_loss3, flow3_3d = self.compute_loss(flow3, label)

        # upsample flow3
        flow3_up = self.upsample(flow3)

        dc2 = self.deconv2(xc3)
        xc2 = self.xconv2(torch.cat((dc2, flow3_up, c2), 1))

        flow2 = torch.tanh(self.flow2(xc2))  # flow2 : stacks of (56,56) w/ L=16
        flow_loss2, smooth_loss2, flow2_3d = self.compute_loss(flow2, label)

        flow_loss = torch.cat(
            [x.unsqueeze(0) for x in [flow_loss4, flow_loss3, flow_loss2]], 0
        ).mean()
        smooth_loss = torch.cat(
            [x.unsqueeze(0) for x in [smooth_loss4, smooth_loss3, smooth_loss2]], 0
        ).mean()

        flow_predictions = nn.Upsample(scale_factor=(1,2,2))(flow2_3d)

        return flow_loss, smooth_loss, flow_predictions
