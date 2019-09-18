import os
from models import resnet
import torch
from torch import nn
from utils.parallel import DataParallelModel
import models.regression_model as regression_model
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models


def generate_backbone(opt):

    net = None

    if opt.backbone == '3D-resnet':
        if opt.model_depth == 50:
            net = resnet.resnet50(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

        elif opt.model_depth == 101:
            net = resnet.resnet101(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

        else:
            ValueError("Invalid model depth")

    # other models...
    if opt.backbone == '2D-resnet':
        if opt.mode == "preprocess__feature":
            if opt.model_depth == 50:
                net = models.resnet50(pretrained=True)

            elif opt.model_depth == 101:
                net = models.resnet101(pretrained=True)

            elif opt.model_depth == 152:
                net = models.resnet152(pretrained=True)

            for param in net.parameters():
                param.requires_grad = False
        else:
            return net

    # if pre-trained modelfile exists...
    if opt.pretrained_path:
        print(f"Load pretrained model from {opt.pretrained_path}...")
        net = nn.DataParallel(net)

        # laod pre-trained model
        pretrain = torch.load(opt.pretrained_path,
                              map_location=torch.device('cpu'))
        net.load_state_dict(pretrain['state_dict'])

    # net will be uploaded to GPU later..

    return net


def generate_regression_model(backbone, opt):

    net = None

    if opt.backbone == "3D-resnet":

        if opt.model_arch == 'SPP':
            net = regression_model.SpatialPyramid(
                backbone=backbone, dilation_config=(1, 6, 12, 18, 24),
                num_units=opt.num_units, n_factors=opt.n_factors,
                kernel_size=3, drop_rate=opt.drop_rate)

        if opt.model_arch == "HPP":
            if opt.merge_type == 'addition':
                net = regression_model.HPP_Addition_Net(
                    num_units=opt.num_units, n_factors=opt.n_factors,
                    backbone=backbone, drop_rate=opt.drop_rate,
                    n_groups=opt.n_groups)
            elif opt.merge_type == '1x1_C':
                net = regression_model.HPP_1x1_Net(
                    num_units=opt.num_units, n_factors=opt.n_factors,
                    backbone=backbone, drop_rate=opt.drop_rate,
                    attention=opt.attention,
                    n_groups=opt.n_groups)

        elif opt.model_arch == 'naive':
            net = regression_model.Naive_Flatten_Net(num_units=opt.num_units,
                                                     n_factors=opt.n_factors,
                                                     backbone=backbone)
    elif opt.backbone == "2D-resnet":
        if opt.model_arch == 'DeepFFT':
            net = regression_model.DeepFFT(num_feats=2048,
                                           n_factors=opt.n_factors,
                                           num_freq=100,
                                           drop_rate=opt.drop_rate)

    # Enable GPU model & data parallelism
    if opt.multi_gpu:
        net = DataParallelModel(net, device_ids=eval(opt.device_ids + ',', ))

    net.cuda()

    return net


def init_state(opt):
    # define backbone
    backbone = generate_backbone(opt)

    # define regression model
    net = generate_regression_model(backbone, opt)

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer = optim.SGD(
        net.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=opt.lr_patience)

    return net, optimizer, scheduler


def load_trained_ckpt(opt, net):

    ckpt_dir = os.path.join(opt.ckpt_dir,
                            '_'.join(filter(lambda x: x != '',
                                            [opt.model_arch,
                                                opt.merge_type,
                                                opt.arch])))
    model_path = os.path.join(ckpt_dir, 'save_' + opt.test_epoch + '.pth')
    print(f"Load trained model from {model_path}...")

    # laod pre-trained model
    pretrain = torch.load(model_path)
    net.load_state_dict(pretrain['state_dict'])

    return net
