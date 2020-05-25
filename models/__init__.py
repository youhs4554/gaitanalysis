import torch
import torch.nn as nn

from .agnets import *
from .baseline import *
from .default_predictor import *
from .guidenet import *
from .utils import *
from manifest.target_columns import BASIC_GAIT_PARAMS


def generate_network(opt, n_outputs=2, target_transform=None):
    # define backbone
    backbone, backbone_dims = generate_backbone(opt.backbone)

    if opt.model_arch == 'DefaultAGNet':
        # def default_agnet(opt, backbone, backbone_dims, n_outputs, load_pretrained_agnet=False, target_transform=None):
        net = default_agnet(
            opt, backbone, backbone_dims, n_outputs, target_transform)
    elif opt.model_arch == 'RegionalAGNet':
        # def regional_agnet(opt, backbone, backbone_dims, n_outputs, load_pretrained_agnet=False, target_transform=None):
        net = regional_agnet(
            opt, backbone, backbone_dims, n_outputs, target_transform)
    elif opt.model_arch == 'ConcatenatedAGNet':
        # only for gait data
        pretrained_agnet = default_agnet(
            opt, backbone, backbone_dims, len(BASIC_GAIT_PARAMS), load_pretrained_agnet=True)
        net = concatenated_agnet(
            opt, backbone, backbone_dims, n_outputs-len(BASIC_GAIT_PARAMS), pretrained_agnet, target_transform)
    elif opt.model_arch == 'FineTunedConvNet':
        net = fine_tuned_convnet(
            opt, backbone, backbone_dims, n_outputs, target_transform)
    else:
        raise ValueError('Arch {} is not supported'.format(opt.model_arch))

    # Enable GPU model & data parallelism
    if opt.multi_gpu:
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        net = net.cuda()

    return net
