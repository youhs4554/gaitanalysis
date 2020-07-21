import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce
from .utils import freeze_layers, generate_backbone
from .guidenet import GuideNet
from .default_predictor import DefaultPredictor
import copy

__all__ = ["GeneralizedAGNet"]


class GeneralizedAGNet(nn.Module):
    def __init__(self, guider, predictor, target_transform=None):

        super(GeneralizedAGNet, self).__init__()

        self.guider = guider
        self.predictor = predictor
        self.target_transform = target_transform

    def forward(
        self, *inputs, targets=None, averaged=None, return_intermediate_feats=False
    ):

        loss_dict = {}
        tb_dict = {}

        x, guide_loss_dict, guide_tb_dict = self.guider(*inputs)

        # classifier
        out, predictor_loss_dict = self.predictor(x, targets, averaged)

        loss_dict.update(guide_loss_dict)
        loss_dict.update(predictor_loss_dict)

        tb_dict.update(guide_tb_dict)

        if self.target_transform is not None:
            out = self.target_transform(out)

        if return_intermediate_feats:
            return out, x, loss_dict, tb_dict

        return out, loss_dict, tb_dict
