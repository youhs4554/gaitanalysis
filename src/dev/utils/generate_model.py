import os
import sys
from models import resnet
import torch
from torch import nn
from utils.parallel import DataParallelModel, DataParallelCriterion


def generate_backbone(opt):

    net = None

    if opt.backbone=='resnet':
        if opt.model_depth==50:
            net = resnet.resnet50(sample_size=opt.sample_size, sample_duration=opt.sample_duration)

        elif opt.model_depth == 101:
            net = resnet.resnet101(sample_size=opt.sample_size, sample_duration=opt.sample_duration)

        else:
            ValueError("Invalid model depth")

    # other models...



    # if pre-trained modelfile exists...
    if opt.pretrained_path:
        print(f"Load pretrained model from {opt.pretrained_path}...")
        net = nn.DataParallel(net)

        # laod pre-trained model
        pretrain = torch.load(opt.pretrained_path, map_location=torch.device('cpu'))
        net.load_state_dict(pretrain['state_dict'])

    # net will be uploaded to GPU later..

    return net
