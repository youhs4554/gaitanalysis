import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce

from torch.nn.modules.batchnorm import BatchNorm3d
from .utils import freeze_layers, generate_backbone
from .guidenet import AttentionMapGuideNet, MaskGuideNet
from .default_predictor import DefaultPredictor
import copy

__all__ = [
    'GeneralizedAGNet'
]


class GeneralizedAGNet(nn.Module):
    def __init__(self, backbone, mask_guider, predictor, freeze_backbone=False, target_transform=None):

        super(GeneralizedAGNet, self).__init__()

        self.backbone = backbone
        self.mask_guider = mask_guider
        self.predictor = predictor
        self.target_transform = target_transform
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        """
        Override the default train() to freeze the layers except for the top layers
        :return:
        """
        super(GeneralizedAGNet, self).train(mode)
        if self.freeze_backbone:
            for name, m in self.named_modules():
                if "backbone" in name:
                    if isinstance(m, (nn.Conv3d, nn.BatchNorm3d)):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        if m.bias is not None:
                            m.bias.requires_grad = False

    def forward(self, *inputs, targets=None, return_intermediate_feats=False, enable_tsn=False):
        images, masks = inputs
        batch_size, nclips, *cdhw = images.size()

        if enable_tsn:
            images = images.view(-1, *cdhw)
            masks = masks.view(-1, 1, *cdhw[1:])

        device = images.device

        # backbone features
        feats = self.backbone(images)

        loss_dict = {}

        head_out, modified_mask_logits, mask_guide_loss_dict = self.mask_guider(
            feats, masks)

        # class_scores, modified_masks, mask_guide_loss_dict = self.mask_guider(
        #     feats, masks)

        # logits
        logits, predictor_loss_dict = self.predictor(
            head_out, targets, enable_tsn, batch_size)

        # losses
        loss_dict.update(mask_guide_loss_dict)
        loss_dict.update(predictor_loss_dict)
        if self.target_transform is not None:
            logits = self.target_transform(logits)

        if return_intermediate_feats:
            return logits, modified_mask_logits, loss_dict

        return logits, loss_dict
