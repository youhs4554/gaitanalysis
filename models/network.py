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
from .MAME import MAME, MotionGuidedNLBlock

__all__ = ["MameNet"]


class LayerConfig(object):
    # TODO. will be managed in cfg file (.json/.yaml)
    dims = {"r3d": [64, 64, 128, 256, 512], "i3d": [64, 256, 512, 1024, 2048]}
    mame_loc = {"layer4":3, "layer3": 3, "layer2": 2}


class MameNet(nn.Module, LayerConfig):
    def __init__(self, backbone, n_class=None):
        super().__init__()

        # detach fc/avgpool
        del backbone.fc
        del backbone.avgpool

        # freeze_layers([backbone.stem, backbone.layer1])
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
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)

        self.layers = nn.ModuleDict()

        # n of MAME block
        pos = ord('a')
        for ix, ((name, backbone_layer), feats_dim) in enumerate(
            zip(backbone.named_children(), self.dims["r3d"])
        ):
            if name in self.mame_loc:
                layer_seq = [(f"backbone_{name}", backbone_layer), 
                             (f"block_{chr(pos)}", MAME_Block(feats_dim, layers=self.mame_loc[name]))
                             ]
                self.layers[f"MAME_{chr(pos)}"] = nn.Sequential(OrderedDict(layer_seq))
                pos += 1
            else:
                self.layers[f"backbone_{name}"] = backbone_layer

    def forward(self, video, mask):
        loss_dict = defaultdict(float)
        tb_dict = defaultdict(float)

        x = video
        for name, layer in self.layers.items():
            if not name.startswith("MAME"):
                # simple bakcbone forward
                x = layer(x)
            else:
                # last MAME block always returns both x and mask_pred
                x, mask_pred = layer(x)
                
                # downsample mask_label
                mask_label = F.adaptive_avg_pool3d(mask, mask_pred.shape[2:])

                # compute loss & a prediction sample to visualize
                mask_loss = nn.BCELoss()(mask_pred, mask_label)
                sample_mask_pred = mask_pred[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                sample_mask_label = mask_label[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)

                mask_predictions_status = torch.cat((sample_mask_pred, sample_mask_label), 0)

                probe_name = [ n for n,_ in layer.named_children() ][-1]

                loss_dict["mask_loss"] += mask_loss / len(self.mame_loc)
                tb_dict[f"mask_prediction_status_at_{probe_name}"] = mask_predictions_status
        
        return x, loss_dict, tb_dict


class MAME_Block(nn.Module):
    def __init__(self, feats_dim, layers=1):
        super().__init__()
        self.mask_layer = nn.Conv3d(feats_dim, 1, kernel_size=1)
        self.mame_modules = nn.ModuleList([ MAMEModule(feats_dim) for _ in range(layers) ])

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     module_list = [ self.mask_layer ] + list(self.mame_modules.modules())
    #     for m in module_list:
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out',
    #                                     nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        mask_pred = self.mask_layer(x)
        mask_pred = torch.sigmoid(mask_pred)  # (b,1,t,h,w)

        identity = x

        for module in self.mame_modules:
            x = module(x, mask_pred)

        x += identity
        x = torch.relu(x)

        return x, mask_pred


class MAMEModule(nn.Module):
    def __init__(self, feats_dim=256):

        super(MAMEModule, self).__init__()
        self.inter = inter = feats_dim // 4

        self.enc = nn.Conv3d(feats_dim, inter, kernel_size=1)
        self.dec = nn.Conv3d(inter, feats_dim, kernel_size=1)
        
        # MAME : Mask Augmented Motion Encoding module
        self.Mame= MAME(in_channels=inter)

    def forward(self, x, mask_pred):
        # x : 3d features from a prev layer(or block)
        # mask_pred : downsampled human mask predicted from mask_layer
        identity = x

        mask_pred = mask_pred.repeat(1,self.inter,1,1,1)

        x = self.enc(x)
        x = self.Mame(x, mask_pred)
        x = self.dec(x)

        x += identity
        x = torch.relu(x)

        return x
