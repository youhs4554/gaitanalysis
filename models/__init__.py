import torch
import torch.nn as nn

from .stcnet import *
from .baseline import *
from .default_predictor import *
from .utils import *
from cfg.target_columns import BASIC_GAIT_PARAMS


def generate_network(opt, n_outputs=2, target_transform=None):
    # define backbone
    backbone, backbone_dims = generate_backbone(opt.backbone)

    if opt.model_arch == "STCNet":
        net = stcnet(opt, backbone, backbone_dims, n_outputs, target_transform)
    elif opt.model_arch == "ConcatenatedSTCNet":
        import copy

        backbone_clone = copy.deepcopy(backbone)

        # only for gait data
        pretrained_stcnet = stcnet(
            opt, backbone, backbone_dims, len(BASIC_GAIT_PARAMS), load_pretrained=True
        )
        net = concatenated_stcnet(
            opt,
            backbone_clone,
            backbone_dims,
            pretrained_stcnet,
            n_outputs - len(BASIC_GAIT_PARAMS),
            target_transform,
        )
    elif opt.model_arch == "FineTunedConvNet":
        net = fine_tuned_convnet(
            opt, backbone, backbone_dims, n_outputs, target_transform
        )
    else:
        raise ValueError("Arch {} is not supported".format(opt.model_arch))

    return net
