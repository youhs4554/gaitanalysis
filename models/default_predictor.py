import numpy as np
from models.losses import FocalLoss
import torch
from torch import nn
import random

__all__ = [
    'DefaultPredictor'
]


class DefaultPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task='regression', class_weight=None, include_classifier=True, aux=False):
        super(DefaultPredictor, self).__init__()
        if include_classifier:
            self.classifier_1 = nn.Linear(n_inputs, n_outputs)
            nn.init.xavier_uniform_(self.classifier_1.weight)
            # self.classifier_1.weight.data.normal_(0, 0.01)
            self.classifier_1.bias.data.fill_(0.0)
            if n_outputs == 2 and class_weight is not None:
                if self.classifier_1.bias is not None:
                    print(
                        "#################### init bias for imb dataset #########################")
                    w0, w1 = class_weight.tolist()
                    nn.init.constant(self.classifier_1.bias, np.log(w1/w0))
        else:
            self.classifier_1 = nn.Identity()

        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.class_weight = class_weight
        self.include_classifier = include_classifier
        self.aux = aux

        if task == 'regression':
            self.criterion = nn.SmoothL1Loss()
        elif task == 'classification':
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)
            # self.criterion = FocalLoss(weight=class_weight, gamma=2)

    def forward(self, *inputs):
        x, targets, enable_tsn, batch_size = inputs

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.classifier_1(x)

        if enable_tsn:
            # consensus
            out = out.view(
                batch_size, -1, out.size(-1)).mean(1)

        if targets is None:
            return out, 0.0

        dict_key = self.task + "_loss"

        rel_weight = 1.0
        if self.aux:
            dict_key = "aux_" + dict_key
            rel_weight = 0.3

        loss_dict = {
            dict_key: rel_weight * self.criterion(out, targets)
        }

        return out, loss_dict
