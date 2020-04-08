import os
import time
from models import resnet
import torch
from torch import nn
from utils.parallel import DataParallelModel, DataParallelCriterion
import models.regression_model as regression_model
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn.utils as torch_utils
import torch.nn.functional as F
import pytorch_warmup as warmup  # for LR warmup
from functools import reduce


class FocalLoss(nn.Module):

    def __init__(self, gamma=2., balance_param=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.balance_param = balance_param

    def forward(self, input_tensor, target_tensor):
        logpt = -F.cross_entropy(input_tensor, target_tensor, reduction='sum')
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt

        focal_loss *= self.balance_param

        return focal_loss


class MultiScaled_BCELoss(nn.Module):
    def __init__(self, n_scales):
        super().__init__()
        self.n_scales = n_scales
        self.loss_func = nn.BCELoss(reduction='sum')

    def forward(self, input, target):
        l = []
        for i in range(self.n_scales):
            shapes = input[i].size()[2:]
            target = F.interpolate(target,
                                   size=shapes)
            l.append(self.loss_func(
                input[i], target) / reduce(lambda x, y: x*y, shapes))

        return torch.stack(l).mean()


def generate_backbone(opt, pretrained=True):

    net = None

    if opt.backbone == '3D-resnet':
        if opt.model_depth == 18:
            net = resnet.resnet18(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration, shortcut_type='A')

        elif opt.model_depth == 34:
            net = resnet.resnet34(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration, shortcut_type='A')

        elif opt.model_depth == 50:
            net = resnet.resnet50(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration, shortcut_type='B')

        elif opt.model_depth == 101:
            net = resnet.resnet101(
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration, shortcut_type='B')

        else:
            ValueError("Invalid model depth")

    # other models...
    if opt.backbone == '2D-resnet':
        if opt.mode == "preprocess__feature":
            if opt.model_depth == 50:
                net = models.resnet50(pretrained=pretrained)

            elif opt.model_depth == 101:
                net = models.resnet101(pretrained=pretrained)

            elif opt.model_depth == 152:
                net = models.resnet152(pretrained=pretrained)

            for param in net.parameters():
                param.requires_grad = False
        else:
            return net

    if opt.backbone == 'r2plus1d_18':
        net = models.video.r2plus1d_18(pretrained=pretrained)

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

    criterion = nn.MSELoss()

    if opt.backbone == "3D-resnet":

        if opt.model_arch == 'SPP':
            net = regression_model.SpatialPyramid(
                backbone=backbone, dilation_config=(1, 6, 12, 18, 24),
                num_units=opt.num_units, n_factors=opt.n_factors,
                kernel_size=3, drop_rate=opt.drop_rate)

        elif opt.model_arch == "HPP":
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
                                                     backbone=backbone,
                                                     drop_rate=opt.drop_rate)

    elif opt.backbone == "2D-resnet":
        if opt.model_arch == 'DeepFFT':
            net = regression_model.DeepFFT(num_feats=2048,
                                           n_factors=opt.n_factors,
                                           num_freq=100,
                                           drop_rate=opt.drop_rate)
    elif opt.backbone == 'r2plus1d_18':
        if opt.model_arch == 'AGNet':
            pretrained_agnet = regression_model.AGNet_Mean(
                backbone, hidden_size=512, out_size=16, drop_rate=0.0, freeze=True)
            pretrained_agnet = load_pretrained_ckpt(opt,
                                                    pretrained_agnet)

            net = regression_model.AGNet(
                pretrained_agnet,
                backbone, hidden_size=1024, out_size=4)
        elif opt.model_arch == 'AGNet-pretrain':
            net = regression_model.AGNet_Mean(backbone, hidden_size=512,
                                              out_size=16, drop_rate=0.2)
        elif opt.model_arch == 'GuidelessNet':
            net = regression_model.GuidelessNet(backbone, hidden_size=512,
                                                out_size=16, drop_rate=0.2)

        criterion1 = nn.SmoothL1Loss(reduction='sum')
        criterion2 = MultiScaled_BCELoss(n_scales=4)

    # Enable GPU model & data parallelism
    if opt.multi_gpu:
        net = DataParallelModel(net, device_ids=eval(
            opt.device_ids + ',', )).cuda()

        criterion1 = DataParallelCriterion(
            criterion1, device_ids=eval(opt.device_ids + ",")).cuda()

        criterion2 = DataParallelCriterion(
            criterion2, device_ids=eval(opt.device_ids + ",")).cuda()

    else:
        net = net.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    return net, criterion1, criterion2


def generate_classification_model(backbone, opt):
    if opt.backbone == 'r2plus1d_18':
        if opt.model_arch == 'AGNet-pretrain':
            net = regression_model.AGNet_Mean(backbone, hidden_size=512,
                                              out_size=opt.n_class, drop_rate=0.5)
        elif opt.model_arch == 'GuidelessNet':
            net = regression_model.GuidelessNet(backbone, hidden_size=512,
                                                out_size=opt.n_class, drop_rate=0.5)

        # criterion1 = FocalLoss()
        criterion1 = nn.CrossEntropyLoss(reduction='sum')
        criterion2 = MultiScaled_BCELoss(n_scales=4)

    # Enable GPU model & data parallelism
    if opt.multi_gpu:
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        net = net.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    return net, criterion1, criterion2


def init_state(opt):
    # define backbone
    backbone = generate_backbone(opt)

    if opt.benchmark:
        opt.n_class = 2
        net, criterion1, criterion2 = generate_classification_model(
            backbone, opt)
    else:
        # define regression model
        net, criterion1, criterion2 = generate_regression_model(backbone, opt)

    optimizer = None
    scheduler = None

    if opt.mode != 'test':
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        # optimizer = optim.SGD(
        #     net.parameters(),
        #     lr=opt.learning_rate,
        #     momentum=opt.momentum,
        #     dampening=dampening,
        #     weight_decay=opt.weight_decay,
        #     nesterov=opt.nesterov)

        # import swats

        # optimizer = swats.SWATS(
        #     net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
        # )

        params = [p for p in net.parameters() if p.requires_grad]

        optimizer = optim.AdamW(
            params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
        )
        # optimizer = optim.SGD(
        #     params, lr=opt.learning_rate, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True
        # )

        # optimizer = optim.RMSprop(
        #     net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay,
        #     momentum=opt.momentum,
        # )

        # In orther to avoid gradient exploding, we apply gradient clipping
        torch_utils.clip_grad_norm_(params, opt.max_gradnorm)

        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', verbose=True, patience=opt.lr_patience)
        # _lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        _lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 15], gamma=0.1)
        # _lr_scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=opt.T_max, eta_min=opt.learning_rate*0.003)

        # _warmup_scheduler = warmup.RAdamWarmup(optimizer)
        _warmup_scheduler = warmup.LinearWarmup(
            optimizer, warmup_period=opt.warmup_period)
        _warmup_scheduler.last_step = -1

    return net, criterion1, criterion2, optimizer, _lr_scheduler, _warmup_scheduler


def load_trained_ckpt(opt, net):

    ckpt_dir = os.path.join(opt.ckpt_dir,
                            '_'.join(filter(lambda x: x != '',
                                            [opt.attention_str,
                                             opt.model_arch,
                                             opt.merge_type,
                                             opt.arch,
                                             opt.dataset])))

    model_path = os.path.join(ckpt_dir, f'model_fold-{opt.test_fold}.pth')
    print(f"Load trained model from {model_path}...")

    # laod pre-trained model
    pretrain = torch.load(model_path)
    net.load_state_dict(pretrain['state_dict'])

    return net


def load_pretrained_ckpt(opt, net):

    net = nn.DataParallel(net)

    ckpt_dir = os.path.join(opt.ckpt_dir,
                            '_'.join(filter(lambda x: x != '',
                                            [opt.attention_str,
                                             opt.model_arch+'-pretrain',
                                             opt.merge_type,
                                             opt.arch,
                                             opt.dataset])))

    model_path = os.path.join(ckpt_dir, f'model_fold-{opt.test_fold}.pth')
    print(f"Load pretrained model from {model_path}...")

    # laod pre-trained model
    pretrain = torch.load(model_path,
                          map_location=torch.device('cpu'))
    net.load_state_dict(pretrain['state_dict'])

    return net
