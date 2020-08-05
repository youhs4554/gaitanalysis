import torch
import torch.nn as nn

from .mameNet import *
from .baseline import *
from .default_predictor import *
from .network import *
from .utils import *
from cfg.target_columns import BASIC_GAIT_PARAMS


def generate_network(opt, n_outputs=2, target_transform=None):
    # define backbone
    backbone, backbone_dims = generate_backbone(opt.backbone)

    if opt.model_arch == "DefaultMAMENet":
        # def default_mameNet(opt, backbone, backbone_dims, n_outputs, load_pretrained_mameNet=False, target_transform=None):
        net = default_mameNet(opt, backbone, backbone_dims, n_outputs, target_transform)
    elif opt.model_arch == "RegionalMAMENet":
        # def regional_mameNet(opt, backbone, backbone_dims, n_outputs, load_pretrained_mameNet=False, target_transform=None):
        net = regional_mameNet(opt, backbone, backbone_dims, n_outputs, target_transform)
    elif opt.model_arch == "ConcatenatedMAMENet":
        # only for gait data
        pretrained_mameNet = default_mameNet(
            opt,
            backbone,
            backbone_dims,
            len(BASIC_GAIT_PARAMS),
            load_pretrained_mameNet=True,
        )
        net = concatenated_mameNet(
            opt,
            backbone,
            backbone_dims,
            n_outputs - len(BASIC_GAIT_PARAMS),
            pretrained_mameNet,
            target_transform,
        )
    elif opt.model_arch == "FineTunedConvNet":
        net = fine_tuned_convnet(
            opt, backbone, backbone_dims, n_outputs, target_transform
        )
    else:
        raise ValueError("Arch {} is not supported".format(opt.model_arch))

    return net
