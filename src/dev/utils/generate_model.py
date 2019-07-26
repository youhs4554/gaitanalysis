from models import resnet
import torch
from torch import nn
from utils.parallel import DataParallelModel
import models.regression_model as regression_model
from torch import optim
from torch.optim import lr_scheduler

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

def generate_regression_model(backbone, opt):

    net = None

    if opt.model_arch == "HPP":
        if opt.merge_type=='addition':
            net = regression_model.HPP_Addition_Net(num_units=opt.num_units, n_factors=opt.n_factors, backbone=backbone, drop_rate=opt.drop_rate,
                                                     n_groups=opt.n_groups)
        elif opt.merge_type == '1x1_C':
            # define regression model
            net = regression_model.HPP_1x1_Net(num_units=256, n_factors=15, backbone=backbone, drop_rate=0.0,
                                               attention=opt.attention,
                                               n_groups=3)

    elif opt.model_arch == 'naive':
        net = regression_model.Naive_Flatten_Net(num_units=256,
                                                 n_factors=15,
                                                 backbone=backbone)

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
        optimizer, 'min', patience=opt.lr_patience)

    return net, optimizer, scheduler