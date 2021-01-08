import torch
import torch.nn as nn
import torchvision.models

__all__ = [
    'generate_backbone',
    'freeze_layers',
    'init_layers',
    'load_pretrained_ckpt',
]


def load_pretrained_ckpt(net, pretrained_path=''):
    print(f"Load pretrained model from {pretrained_path}...")

    # laod pre-trained model
    pretrain = torch.load(pretrained_path,
                          map_location=torch.device('cpu'))
    net.load_state_dict(pretrain['state_dict'])

    return net


def get_inflated_resnet(net_2d, net_3d):
    c_ix = 0
    bn_ix = 0
    net_2d_conv_layers = [m for m in list(
        net_2d.modules()) if isinstance(m, nn.Conv2d)]
    net_2d_bn_layers = [m for m in list(net_2d.modules())
                        if isinstance(m, nn.BatchNorm2d)]

    for m in net_3d.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data = net_2d_conv_layers[c_ix].weight.data[:, :, None].repeat(
                (1, 1, m.kernel_size[0], 1, 1))
            c_ix += 1
        if isinstance(m, nn.BatchNorm3d):
            m.weight.data = net_2d_bn_layers[bn_ix].weight.data
            bn_ix += 1

    return net_3d


def generate_backbone(opt, pretrained=True):
    net = None
    if opt.backbone in ['mc3_18', 'r2plus1d_18', 'r3d_18']:
        net_init_func = getattr(torchvision.models.video, opt.backbone)
        net = net_init_func(pretrained=pretrained)
        dims = [64, 64, 128, 256, 512]
    elif "r2plus1d_34" in opt.backbone:
        TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"
        MODELS = {
            # Model name followed by the number of output classes.
            "r2plus1d_34_32_ig65m": 359,
            "r2plus1d_34_32_kinetics": 400,
            "r2plus1d_34_8_ig65m": 487,
            "r2plus1d_34_8_kinetics": 400,
        }
        pretrained_data = opt.backbone.split("_")[-1]
        assert pretrained_data in [
            "ig65m", "kinetics"], "Not supported pretrained data {}, Should be 'ig65m' or 'kinetics'".format(pretrained_data)
        duration = "8" if opt.sample_duration <= 16 else "32"
        model_name = f"r2plus1d_34_{duration}_{pretrained_data}"
        net = torch.hub.load(
            TORCH_R2PLUS1D,
            model_name,
            num_classes=MODELS[model_name],
            pretrained=pretrained,
        )
        dims = [64, 64, 128, 256, 512]
    elif opt.backbone == 'inflated':
        resnet_2d = torchvision.models.resnet18(pretrained)
        resnet_3d = torchvision.models.video.r3d_18(pretrained)
        # inflate 2d weights to
        net = get_inflated_resnet(net_2d=resnet_2d, net_3d=resnet_3d)
        dims = [64, 64, 128, 256, 512]

    # net will be uploaded to GPU later..
    if net is None:
        raise ValueError('invalid backbone Type')

    return net, dims


def freeze_layers(layers):
    for child in layers:
        for p in child.parameters():
            p.requires_grad = False


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def init_layers(layers):
    for child in layers:
        child.apply(init_weights)


def init_mask_layer(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if all(ks == 1 for ks in m.kernel_size):
            # zero weights for only 1x1 conv(last layer)
            m.weight.data.fill_(0.0)

# from datasets.classification.falldown.utils_cv.detection.references.utils import warmup_lr_scheduler
# from datasets import get_data_loader
# from models import generate_network
# import sklearn.metrics
# import warnings
# import os
# import json
# import utils.engine
# import torch.optim as optim
# import torch.nn as nn
# import argparse
# import torch
# import numpy as np
# import random


# def set_seed(seed):
#     """
#     For seed to some modules.
#     :param seed: int. The seed.
#     :return:
#     """
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# warnings.filterwarnings("ignore")


# class Opt():
#     def __init__(self):
#         pass


# opt = Opt()

# paser = argparse.ArgumentParser()
# paser.add_argument("--cfg_file",
#                    default="",
#                    help="path to configuration file")
# paser.add_argument("--fold", type=int)

# args = paser.parse_args()

# if not os.path.exists(args.cfg_file):
#     raise ValueError(
#         "Configuration file {} is not exist...".format(args.cfg_file))

# with open(args.cfg_file, 'r') as f:
#     opt.__dict__ = json.load(f)

# opt.cfg_file = os.path.splitext(os.path.basename(args.cfg_file))[0]
# opt.fold = args.fold


# def train_one_fold(fold, metrics=['f1-score', 'accuracy', 'ap', 'roc_auc']):
#     # Load data
#     train_loader, test_loader, target_transform, n_outputs = get_data_loader(
#         opt, fold=fold)

#     if opt.model_arch == 'RegionalAGNet':
#         n_outputs = n_outputs + 1

#     # Init model
#     model = generate_network(opt, n_outputs=n_outputs,
#                              target_transform=target_transform)

#     # re-seeding after model initialization
#     set_seed(0)

#     # Define optimizer & schedulers
#     params = [p for p in model.parameters() if p.requires_grad]
#     if opt.backbone == "r2plus1d_18":
#         from ranger import Ranger  # this is from ranger.py
#         optimizer = Ranger(params, lr=opt.learning_rate,
#                            weight_decay=opt.weight_decay, k=10)
#         lr_scheduler = None
#         warmup_scheduler = None
#     else:
#         optimizer = optim.SGD(params, lr=opt.learning_rate,
#                               momentum=0.9,
#                               weight_decay=opt.weight_decay)

#         lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=len(train_loader)*opt.n_iter, eta_min=1e-5)

#         # lr_scheduler = None
#         warmup_factor = 1. / 1000
#         warmup_iters = min(1000, 5*len(train_loader))
#         warmup_scheduler = warmup_lr_scheduler(
#             optimizer, warmup_iters, warmup_factor)

#     # Define NeuralNetwork Wrapper Class
#     net = utils.engine.VideoClassifier(model, optimizer,
#                                        n_folds=opt.n_folds,
#                                        fold=fold,
#                                        lr_scheduler=lr_scheduler,
#                                        warmup_scheduler=warmup_scheduler)

#     net.train(train_loader, test_loader,
#               n_epochs=opt.n_iter, validation_freq=len(train_loader),
#               multiple_clip=opt.multiple_clip, metrics=metrics,
#               save_dir=os.path.join(opt.ckpt_dir, opt.cfg_file))


# if __name__ == "__main__":

#     # for reproducibility
#     set_seed(0)

#     # LOO cross-validation loop
#     train_one_fold(opt.fold, metrics={
#         'f1-score': (lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred), False),
#         'accuray': (lambda y_true, y_pred: sklearn.metrics.accuracy_score(y_true, y_pred), False),
#         'roc_auc': (lambda y_true, y_score: sklearn.metrics.roc_auc_score(y_true, y_score), True),
#         'ap': (lambda y_true, y_score: sklearn.metrics.average_precision_score(y_true, y_score), True),
#         # custom callbacks
#         'sensitivity': (lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, pos_label=1), False),
#         'specificity': (lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, pos_label=0), False),
#         'precision': (lambda y_true, y_pred: sklearn.metrics.precision_score(y_true, y_pred, pos_label=1), False)

#     })
