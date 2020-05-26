from datasets import get_data_loader
from models import generate_network
import sklearn.metrics
import warnings
import os
import json
import utils.engine
import torch.optim as optim
import torch.nn as nn
import pytorch_warmup as warmup  # for LR warmup
import argparse
import torch
import numpy as np
import random


def set_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


warnings.filterwarnings("ignore")


class Opt():
    def __init__(self):
        pass


opt = Opt()

paser = argparse.ArgumentParser()
paser.add_argument("--cfg_file",
                   default="",
                   help="path to configuration file")
args = paser.parse_args()

if not os.path.exists(args.cfg_file):
    raise ValueError(
        "Configuration file {} is not exist...".format(args.cfg_file))

with open(args.cfg_file, 'r') as f:
    opt.__dict__ = json.load(f)


def train_one_fold(fold, metrics=['f1-score', 'accuracy', 'ap', 'roc_auc']):
    # for reproductivity
    set_seed(0)

    # Load data
    train_loader, test_loader, target_transform, n_outputs = get_data_loader(
        opt, fold=fold)

    if opt.model_arch == 'RegionalAGNet':
        n_outputs = n_outputs + 1

    # Init model
    model = generate_network(opt, n_outputs=n_outputs,
                             target_transform=target_transform)

    # Define optimizer & schedulers
    params = [p for p in model.parameters() if p.requires_grad]
    from ranger import Ranger  # this is from ranger.py
    optimizer = Ranger(params, lr=opt.learning_rate,
                       weight_decay=opt.weight_decay, k=10)

    torch.nn.utils.clip_grad_norm_(params, opt.max_gradnorm)

    # optimizer = optim.SGD(params, lr=opt.learning_rate,
    #                       momentum=0.9,
    #                       weight_decay=opt.weight_decay)

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[opt.n_iter*0.25, opt.n_iter*0.75], gamma=1.0)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                     T_max=int(len(train_loader)*opt.n_iter*0.25))

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max',
        factor=0.1,
        patience=5, verbose=True)

    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=10, eta_min=1e-5)

    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # lr_scheduler = None
    warmup_scheduler = None

    # Define NeuralNetwork Wrapper Class
    net = utils.engine.VideoClassifier(model, optimizer,
                                       n_folds=opt.n_folds,
                                       lr_scheduler=lr_scheduler,
                                       warmup_scheduler=warmup_scheduler)

    net.train(train_loader, test_loader,
              n_epochs=opt.n_iter, validation_freq=len(train_loader),
              multiple_clip=opt.multiple_clip, metrics=metrics,
              save_dir=os.path.join(opt.ckpt_dir, opt.model_indicator))

# TODO. Refactoring model trainer, pytorch-lightning!!!!
# https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09


for fold in range(1, opt.n_folds+1):
    fold = 1
    # LOO cross-validation loop
    train_one_fold(fold, metrics={
        'f1-score': (lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred), False),
        'accuray': (lambda y_true, y_pred: sklearn.metrics.accuracy_score(y_true, y_pred), False),
        'roc_auc': (lambda y_true, y_score: sklearn.metrics.roc_auc_score(y_true, y_score), True),
        'ap': (lambda y_true, y_score: sklearn.metrics.average_precision_score(y_true, y_score), True),
        # custom callbacks
        'sensitivity': (lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, pos_label=1), False),
        'specificity': (lambda y_true, y_pred: sklearn.metrics.recall_score(y_true, y_pred, pos_label=0), False),
        'precision': (lambda y_true, y_pred: sklearn.metrics.precision_score(y_true, y_pred, pos_label=1), False)

    })
    break
