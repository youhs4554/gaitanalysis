import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce
from .utils import freeze_layers, generate_backbone
from .default_predictor import DefaultPredictor
import copy

__all__ = ["Net"]


class Net(nn.Module):
    def __init__(self, stc_network, predictor, target_transform=None):

        super(Net, self).__init__()

        self.stc_network = stc_network
        self.predictor = predictor
        self.target_transform = target_transform

    def forward(
        self, *inputs, targets=None, lambda_=None, return_intermediate_feats=False
    ):

        loss_dict = {}
        tb_dict = {}

        x, feats_dict, stc_loss_dict, stc_tb_dict = self.stc_network(*inputs)

        # classifier
        out, predictor_loss_dict = self.predictor(x, targets, lambda_)

        if targets is None:
            return out, feats_dict

        loss_dict.update(stc_loss_dict)
        loss_dict.update(predictor_loss_dict)

        tb_dict.update(stc_tb_dict)

        if self.target_transform is not None:
            out = self.target_transform(out)

        if return_intermediate_feats:
            return out, x, loss_dict, tb_dict

        return out, loss_dict, tb_dict
