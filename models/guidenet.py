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
from .motion_guided_conv3d import TinyMotionNet

__all__ = ["GuideNet"]


class LayerConfig(object):
    # TODO. will be managed in cfg file (.json/.yaml)
    dims = {"r3d": [64, 64, 128, 256, 512]}
    guide_loc = {"layer1": 0, "layer2": 1, "layer3": 2}


class GuideNet(nn.Module, LayerConfig):
    def __init__(self, backbone, n_class=None):
        super().__init__()

        # detach fc/avgpool
        del backbone.fc
        del backbone.avgpool

        freeze_layers([backbone.stem,backbone.layer1])
        for ix, ((name, backbone_layer), feats_dim) in enumerate(
            zip(backbone.named_children(), self.dims["r3d"])
        ):
            if name in self.guide_loc:
                self.add_module(
                    f"Block_{name}_guided", GuideBlock(backbone_layer, feats_dim)
                )
            else:
                self.add_module(f"Block_{name}", backbone_layer)

        # motion_net : predicts optical flows at various resolutions
        self.motion_net = TinyMotionNet(n_frames=16)
        
    def forward(self, images, masks, flows):
        loss_dict = defaultdict(float)
        tb_dict = defaultdict(float)

        flow_loss, smooth_loss, flow_predictions = self.motion_net(images, flows)

        # losses for optical flow generation
        loss_dict["flow_loss"] = flow_loss
        loss_dict["smooth_loss"] = smooth_loss

        # TODO. visualize flow_predictions?
        # tb_dict["flows_pred"] = flow_predictions[-1]

        blocks = [
            (name, getattr(self, name))
            for name, _ in self.named_children()
            if name.startswith("Block_")
        ]
        # except last frame
        x = images[:, :, :-1]
        masks = masks[:, :, :-1]

        for name, block in blocks:
            if name.endswith("_guided"):
                x, mask_loss, mask_predictions_status = block(
                    x, masks, flow_predictions[self.guide_loc[name.split("_")[1]]]
                )
                loss_dict["mask_loss"] += mask_loss / len(self.guide_loc)
                tb_dict[f"mask_prediction_status_at_{name}"] = mask_predictions_status
            else:
                x = block(x)

        return x, loss_dict, tb_dict


class GuideBlock(nn.Module):
    def __init__(self, backbone_layer, feats_dim, temporal_stride=1):
        super().__init__()
        self.backbone_layer = backbone_layer
        self.guide_module = GuideModule(feats_dim)

    def forward(self, x, mask, flow_pred):
        x = self.backbone_layer(x)

        # downsample mask
        mask_downsampled = F.adaptive_avg_pool3d(mask, x.shape[2:])

        # downsample flow_pred
        flow_pred_downsampled = F.adaptive_avg_pool3d(flow_pred, x.shape[2:])

        x, mask_loss, mask_activation_sample, mask_sample = self.guide_module(
            x, mask_downsampled, flow_pred_downsampled
        )

        mask_predictions_status = torch.cat((mask_activation_sample, mask_sample), 0)

        return x, mask_loss, mask_predictions_status


class GuideModule(nn.Module):
    def __init__(self, feats_dim=256):

        super(GuideModule, self).__init__()

        # predicts human masks
        self.mask_layer = nn.Conv3d(feats_dim, 1, kernel_size=1, bias=True)
        self.blending_layer = nn.Conv3d(2 + 1, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask, flow_pred):
        # x : 3d features from backbone layer
        # mask : downsampled ground-truth human mask
        # flow_pred : downsampled optical flows predicted from TinyFlowNet
        identity = x

        mask_pred = self.mask_layer(x)
        mask_pred = torch.sigmoid(mask_pred)

        # compute loss & a prediction sample
        mask_loss = nn.BCELoss()(mask_pred, mask)
        mask_activation_sample = mask_pred[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        mask_sample = mask[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)

        concat = torch.cat((flow_pred, mask_pred), 1)
        blended_loc = self.blending_layer(concat)
        blended_loc = torch.sigmoid(blended_loc)

        x = blended_loc * x

        x += identity
        x = self.relu(x)

        return x, mask_loss, mask_activation_sample, mask_sample
