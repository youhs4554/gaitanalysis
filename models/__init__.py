import os
import random
import numpy as np
import torch
import torch.nn as nn

from .agnets import *
from .baseline import *
from .default_predictor import *
from .guidenet import *
from .utils import *
from manifest.target_columns import BASIC_GAIT_PARAMS


def generate_network(opt, n_outputs=2, target_transform=None):
    freeze_backbone = False
    if getattr(opt, "finetuned_backbone", None):
        freeze_backbone = True
        model_filepath = opt.finetuned_backbone

        # use backbone finetuned with each dataset
        backbone = torch.load(model_filepath).backbone
        # freeze_layers(backbone.children())
        # backbone.eval()

        # detach last layer
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone.to(torch.device("cpu"))
        print("### Loaded finetuned backbone from {} ###".format(model_filepath))
        print()

    else:
        # define backbone
        backbone = generate_backbone(opt, pretrained=True)

    dummy_inp = torch.ones(1, 3, opt.sample_duration,
                           opt.sample_size, opt.sample_size)
    dummy_out = backbone(dummy_inp)
    inplanes = dummy_out.size(1)

    if "AGNet" in opt.model_arch:
        # def default_agnet(opt, backbone, inplanes, n_outputs, load_pretrained_agnet=False, target_transform=None):
        net = default_agnet(
            opt, backbone, inplanes, n_outputs, freeze_backbone=freeze_backbone, target_transform=target_transform)
    elif opt.model_arch == 'FineTunedConvNet':
        net = fine_tuned_convnet(
            opt, backbone, inplanes, n_outputs, target_transform)
    else:
        raise ValueError('Arch {} is not supported'.format(opt.model_arch))

    # Enable GPU model & data parallelism if multi-gpu system
    net = (
        nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    )

    if torch.cuda.is_available():
        net = net.cuda()

    return net
