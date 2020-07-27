import torch
from torch import nn
import torch.nn.functional as F
from .losses import MultiScaled_BCELoss, MultiScaled_DiceLoss, BinaryDiceLoss, EpeLoss
from .utils import init_mask_layer, freeze_layers
from collections import OrderedDict, defaultdict
from functools import reduce
import math
import random
from .i3res import inflated_resnet
from .deformable_conv3d import DeformConv3DWrapper
from .motion_guided_conv3d import TinyMotionNet
from .NLBlock import NONLocalBlock3D
import kornia

__all__ = ["GuideNet"]


class LayerConfig(object):
    # TODO. will be managed in cfg file (.json/.yaml)
    dims = {"r3d": [64, 64, 128, 256, 512], "i3d": [64, 256, 512, 1024, 2048]}
    guide_loc = {"layer3": 2, "layer2": 1}


class GuideNet(nn.Module, LayerConfig):
    def __init__(self, backbone, n_class=None):
        super().__init__()

        # detach fc/avgpool
        del backbone.fc
        del backbone.avgpool

        count = 0
        for name, m in backbone.named_modules():
            # if m.__class__.__name__.find("BatchNorm") != -1:
            #     count += 1
            #     if count > 2:
            #         # freeze bn layers of backbone except first layer
            #         m.eval()

            #         m.weight.requires_grad = False
            #         m.bias.requires_grad = False

            if name.endswith("bn3"):
                # init gamma/bias of last bn layers with 0 to use pre-trained models
                # at the beginnning of training
                m.weight.data.fill_(0.0)
                m.bias.data.fill_(0.0)

        # freeze_layers([backbone.stem, backbone.layer1])
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
        self.mask_layer = nn.Conv3d(feats_dim, 1, kernel_size=1)
        self.guide_module = GuideModule(feats_dim)

    def forward(self, x, mask, flow_pred):
        x = self.backbone_layer(x)

        # downsample mask
        mask_downsampled = F.adaptive_avg_pool3d(mask, x.shape[2:])

        # downsample flow_pred
        flow_pred_downsampled = F.adaptive_avg_pool3d(flow_pred, x.shape[2:])
        print("Min : ", x.min(), "Max : ", x.max())
        # predicts human masks
        mask_pred = self.mask_layer(x)
        mask_pred = torch.sigmoid(mask_pred)  # (b,1,t,h,w)

        x = self.guide_module(x, mask_pred, flow_pred_downsampled)

        # compute loss & a prediction sample to visualize
        mask_loss = nn.BCELoss()(mask_pred, mask_downsampled)
        mask_activation_sample = mask_pred[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        mask_sample = mask_downsampled[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)

        mask_predictions_status = torch.cat((mask_activation_sample, mask_sample), 0)

        return x, mask_loss, mask_predictions_status


class GuideModule(nn.Module):
    def __init__(self, feats_dim=256):

        super(GuideModule, self).__init__()

        self.embedding_layer = nn.Conv3d(1, feats_dim, kernel_size=1)

        self.nl_block = NONLocalBlock3D(in_channels=feats_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.9)

    def forward(self, x, mask_pred, flow_pred):
        # x : 3d features from backbone layer
        # mask_pred : downsampled human mask predicted from mask_layer
        # flow_pred : downsampled optical flows predicted from TinyFlowNet

        flow_mag = (flow_pred[:, 0].pow(2) + flow_pred[:, 1].pow(2)).sqrt()
        flow_mag = flow_mag.unsqueeze(1)

        guide = mask_pred * flow_mag
        guide = self.embedding_layer(guide)
        guide = self.relu(guide)

        out = self.nl_block(x, guide)
        out = self.relu(out)

        return out
