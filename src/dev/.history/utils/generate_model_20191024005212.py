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

        elif opt.model_arch == 'SRegression':
            net = regression_model.SRegessionNet(backbone)
            for name, child in net.named_children():
                if name in ['conv2', 'conv3', 'conv4', 'conv5']:
                    for p in child.parameters():
                        p.requires_grad = False

            criterion1 = nn.BCELoss(reduction='sum')
            criterion2 = nn.MSELoss(reduction='mean')

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
        if opt.model_arch == 'SRegression':
            net = regression_model.SRegessionNet(backbone)

            class MultiScaled_BCELoss(nn.Module):
                def __init__(self, n_scales):
                    super().__init__()
                    self.n_scales = n_scales
                    self.loss_func = nn.BCELoss()

                def forward(self, input, target):
                    l = []
                    import visdom
                    import matplotlib.pyplot as plt
                    viz = visdom.Visdom('133.186.162.37')
                    cmap = plt.get_cmap('jet')

                    for i in range(self.n_scales):
                        target = F.interpolate(target,
                                               size=input[i].size()[2:])
                        l.append(self.loss_func(input[i], target))
                        img = input[i][0, 0, 0]
                        img = (img-img.min())/(img.max()-img.min())

                        viz.image(cmap(img.detach().cpu().numpy())[..., :3].transpose(
                            2, 0, 1), opts=dict(height=224, width=224, title=f'seg{i+1} epoch-{20}'))

                    return torch.stack(l).mean()

            criterion1 = nn.MSELoss()
            criterion2 = MultiScaled_BCELoss(n_scales=5)

    # Enable GPU model & data parallelism
    if opt.multi_gpu:
        net = DataParallelModel(net, device_ids=eval(
            opt.device_ids + ',', )).cuda()

        criterion1 = DataParallelCriterion(
            criterion1, device_ids=eval(opt.device_ids + ",")).cuda()

        criterion2 = DataParallelCriterion(
            criterion2, device_ids=eval(opt.device_ids + ",")).cuda()

    return net, criterion1, criterion2


def init_state(opt):
    # define backbone
    backbone = generate_backbone(opt)

    # define regression model
    net, criterion1, criterion2 = generate_regression_model(backbone, opt)

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

    optimizer = optim.Adam(
        net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
    )

    # optimizer = optim.RMSprop(
    #     net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay,
    #     momentum=opt.momentum,
    # )

    # In orther to avoid gradient exploding, we apply gradient clipping
    torch_utils.clip_grad_norm_(net.parameters(), opt.max_gradnorm)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=opt.lr_patience)

    return net, criterion1, criterion2, optimizer, scheduler


def load_trained_ckpt(opt, net):

    ckpt_dir = os.path.join(opt.ckpt_dir,
                            '_'.join(filter(lambda x: x != '',
                                            [opt.attention_str,
                                             opt.model_arch,
                                             opt.merge_type,
                                             opt.arch,
                                             opt.group_str])))

    model_path = os.path.join(ckpt_dir, 'save_' + opt.test_epoch + '.pth')
    print(f"Load trained model from {model_path}...")

    # laod pre-trained model
    pretrain = torch.load(model_path)
    net.load_state_dict(pretrain['state_dict'])

    return net
