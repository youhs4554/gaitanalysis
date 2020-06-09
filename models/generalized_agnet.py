import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce
from .utils import freeze_layers, generate_backbone
from .guidenet import GuideNet
from .default_predictor import DefaultPredictor
import copy

__all__ = [
    'GeneralizedAGNet'
]


class GeneralizedAGNet(nn.Module):
    def __init__(self, guider, predictor, target_transform=None):

        super(GeneralizedAGNet, self).__init__()

        self.guider = guider
        self.predictor = predictor
        self.target_transform = target_transform

        # self.register_forward_hook()

        # TODO. implant register_forward_hook like `ActivationMapProvider`

    # def register_forward_hook(self):
    #     for name, module in self.named_modules():
    #         if module.__class__.__name__ in ['Conv3d', 'BatchNorm3d', 'ReLU', 'Linear']:
    #             print(name, module)

    #     import ipdb
    #     ipdb.set_trace()

    def forward_guider(self, images, masks):
        return self.guider(images, masks)

    def get_activation_set(self, images):
        # TODO. fashion used in `ActivationMapProvider`, init -> forward() -> get_activation_set
        # return all of the activations collections.OrderedDict({"layer1":v1, "layer2":v2, ...})

        # mask = None -> guider returns dummy loss
        feats, *_ = self.forward_guider(images, masks=None)
        return feats

    def forward(self, *inputs, targets=None, averaged=None, return_intermediate_feats=False):

        loss_dict = {}
        x, masks, guide_loss_dict = self.forward_guider(*inputs)

        # model output
        out, predictor_loss_dict = self.predictor(x, targets, masks, averaged)

        loss_dict.update(guide_loss_dict)
        loss_dict.update(predictor_loss_dict)

        if self.target_transform is not None:
            out = self.target_transform(out)

        if return_intermediate_feats:
            return out, x, loss_dict

        return out, loss_dict
