import torch
from torch import nn
import torch.nn.functional as F
from .losses import MultiScaled_BCELoss, MultiScaled_DiceLoss, BinaryDiceLoss
from .utils import init_mask_layer, freeze_layers
from collections import OrderedDict, defaultdict
from functools import reduce
import math
import random
from .i3res import inflated_resnet
from .deformable_conv3d import DeformConv3DWrapper

__all__ = ["GuideNet"]


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma, norm=False):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.norm = norm
        self.centers = float(self.min) + self.delta * (
            torch.arange(self.bins).float() + 0.5
        )

    def forward(self, x):
        bs, dim = x.size()[:2]
        x = x.flatten(0)

        centers = torch.tensor(self.centers, device=x.device)

        x = torch.unsqueeze(x, 0) - torch.unsqueeze(centers, 1)
        x = (
            torch.exp(-0.5 * (x / self.sigma) ** 2)
            / (self.sigma * math.sqrt(math.pi * 2))
            * self.delta
        )
        x = x.view(bs, dim, -1)
        x = x.sum(dim=1)

        if self.norm:
            summ = (x.sum(1) + 0.000001).unsqueeze(1)
            return x / summ

        return x


class GuideBlock(nn.Module):
    def __init__(self, backbone_layer, feats_dim):
        super().__init__()
        self.add_module("backbone_layer", backbone_layer)
        self.add_module("guide_module", GuideModule(feats_dim))

    def forward(self, x, mask):
        loss_dict = defaultdict(float)
        tb_dict = defaultdict(float)

        x = self.backbone_layer(x)
        x, mask_pred = self.guide_module(x)

        mask_downsampled = F.adaptive_avg_pool3d(mask, x.shape[2:])
        loss_dict["mask_loss"] = nn.BCELoss()(mask_pred, mask_downsampled)

        mask_activation_sample = mask_pred[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        mask_true_sample = mask_downsampled[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)

        tb_dict["mask_predictions"] = torch.cat(
            (mask_activation_sample, mask_true_sample), 0
        )

        return x, loss_dict, tb_dict


class LayerConfig(object):
    dims = {"r3d": [64, 64, 128, 256, 512], "inflated-r3d": [64, 256, 512, 1024, 2048]}
    guide_ixs = [4]
    fusion_ix = 4


class GuideNet(nn.Module, LayerConfig):
    def __init__(self, backbone, n_class=None):
        super().__init__()

        appearance_stream = inflated_resnet(arch="resnet50")

        appearance_stream_children = nn.Sequential(
            OrderedDict(
                {name: child for name, child in appearance_stream.named_children()}
            )
        )

        appearance_feature_branch, appearance_main_branch = (
            appearance_stream_children[: self.fusion_ix],
            appearance_stream_children[self.fusion_ix :],
        )

        # override forward method
        appearance_predictor = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(appearance_main_branch.fc.in_features, n_class)
        )

        # detach fc from main branch
        del appearance_main_branch.fc

        # inflate 2D resnet to 3D (use conv4)
        self.appearance_feature_branch = appearance_feature_branch
        # auxiliary classifier affected by appearance feature
        self.appearance_main_branch = appearance_main_branch
        self.appearance_predictor = appearance_predictor

        # detach fc/avgpool
        del backbone.fc
        del backbone.avgpool

        for ix, ((name, backbone_layer), feats_dim) in enumerate(
            zip(backbone.named_children(), self.dims["r3d"])
        ):
            if ix in self.guide_ixs:
                self.add_module(
                    f"Block_{name}_guided", GuideBlock(backbone_layer, feats_dim)
                )
            else:
                self.add_module(f"Block_{name}", backbone_layer)

        # fusion layer
        self.fusion_space = self.dims["r3d"][self.fusion_ix]
        self.branch_space = self.dims["inflated-r3d"][self.fusion_ix - 1]
        self.fusion_layer = nn.Sequential(
            conv1x1(self.fusion_space + self.branch_space, self.fusion_space),
            nn.BatchNorm3d(self.fusion_space),
            nn.ReLU(inplace=True),
        )

    def forward(self, images, masks):
        loss_dict = defaultdict(float)
        tb_dict = defaultdict(float)

        appearance_f = self.appearance_feature_branch(images)

        blocks = [
            (name, getattr(self, name))
            for name, _ in self.named_children()
            if name.startswith("Block_")
        ]
        x = images
        for ix, (name, block) in enumerate(blocks):
            if ix in self.guide_ixs:
                x, _loss_dict, _tb_dict = block(x, masks)

                postfix = f"_at_{name}"

                # update dictionaries
                for k in _loss_dict:
                    loss_dict[k] += _loss_dict[k] / len(self.guide_ixs)
                for k in _tb_dict:
                    tb_dict[k + postfix] = _tb_dict[k]
            else:
                x = block(x)

            if ix == self.fusion_ix:
                x_cat = torch.cat((x, appearance_f), dim=1)
                x = self.fusion_layer(x_cat)

        appearance_f = self.appearance_main_branch(appearance_f).flatten(1)

        # predicts classes based on appearance
        # this will be fused later...
        appearance_pred = self.appearance_predictor(appearance_f)

        return x, appearance_pred, loss_dict, tb_dict


def conv3x3(
    in_planes, out_planes, midplanes=None, stride=1, groups=1, dilation=1, use_bias=True
):
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=use_bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1, use_bias=True):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        groups=groups,
        dilation=dilation,
        bias=use_bias,
    )


class GuideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, thresholds):
        ctx.save_for_backward(x, alpha, thresholds)
        x_out = x.clone()
        x_out[alpha < thresholds] = 0.0

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, thresholds = ctx.saved_tensors
        grad_x = grad_alpha = grad_thresholds = None

        grad_x = grad_output.clone()
        grad_x[alpha < thresholds] = 0.0

        return grad_x, grad_alpha, grad_thresholds


def init_weights(m, nonlinearity):
    def wrapper(m):
        if m.__class__.__name__.find("Conv") != -1:
            if nonlinearity == "relu":
                torch.nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
            elif nonlinearity == "sigmoid":
                torch.nn.init.xavier_uniform_(m.weight)

    return wrapper


class PixelwisePredictor(nn.Sequential):
    def __init__(self, inplanes, acitvation="ReLU"):
        super().__init__()
        self.add_module("conv", conv1x1(inplanes, 1, use_bias=True))
        self.add_module("acitvation", getattr(torch.nn, acitvation)())

    def forward(self, *input):
        return super().forward(*input)


# from torchvision.models.video.resnet import Bottleneck


# class PixelwisePredictor(nn.Sequential):
#     def __init__(self, inplanes, acitvation="ReLU"):
#         super().__init__()
#         expansion = Bottleneck.expansion
#         self.add_module(
#             "conv", Bottleneck(inplanes, inplanes // expansion, conv_builder=conv3x3)
#         )
#         self.add_module("prediction", conv1x1(inplanes, 1, use_bias=True))
#         self.add_module("acitvation", getattr(torch.nn, acitvation)())

#     def forward(self, *input):
#         return super().forward(*input)


class OtsuRegularization(nn.Module):
    def __init__(self,):
        super().__init__()
        # differentiable histogram operation
        self.soft_hist = SoftHistogram(bins=100, min=0, max=1, sigma=1.0)

    def forward(self, x, th):
        hist = self.soft_hist(x)

        bin_centers = torch.tensor(
            [0.0] + self.soft_hist.centers.tolist(), device=x.device
        )

        center_ixs = (bin_centers < th).sum(1)
        ixs = torch.arange(hist.size(1), device=x.device)

        cix0 = (ixs < center_ixs[..., None]).float()
        cix1 = (ixs >= center_ixs[..., None]).float()

        c0 = cix0 * hist
        c1 = cix1 * hist

        w0 = c0.sum(1) / hist.sum(1)
        w1 = c1.sum(1) / hist.sum(1)

        bins = (bin_centers[1:] - bin_centers[:-1]).cumsum(0)

        mu0 = (c0 * bins).sum(1) / c0.sum(1)
        mu1 = (c1 * bins).sum(1) / c1.sum(1)

        var0 = ((c0 - mu0.unsqueeze(1)) * bins).pow(2).sum(1) / c0.sum(1)
        var1 = ((c1 - mu1.unsqueeze(1)) * bins).pow(2).sum(1) / c1.sum(1)

        return (w0 * var0 + w1 * var1).mean()


class GuideModule(nn.Module):
    def __init__(self, feats_dim=256):

        super(GuideModule, self).__init__()

        # predicts human masks
        self.mask_layer = PixelwisePredictor(feats_dim, acitvation="Sigmoid")
        self.warp_conv = DeformConv3DWrapper(feats_dim, feats_dim, kernel_size=3)
        self.warp_bn = nn.BatchNorm3d(feats_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x : 3d features from backbone layer
        identity = x

        # warp feature
        warp = self.warp_conv(x)
        warp = self.warp_bn(warp)
        warp = self.dropout(warp)

        mask_pred = self.mask_layer(x)

        # multiply learnable_mask with warped features
        x = mask_pred * warp

        x += identity
        x = self.relu(x)

        return x, mask_pred
