import torch
import torch.nn as nn
from .generalized_agnet import GeneralizedAGNet
from .guidenet import AttentionMapGuideNet, MaskGuideNet
from .default_predictor import DefaultPredictor
from .triplet_predictor import TripletPredictor
from .regional_predictor import RegionalPredictor
from .utils import BackboneHelper, freeze_layers, generate_backbone, load_pretrained_ckpt
import copy

__all__ = [
    'DefaultAGNet',
    'default_agnet',
]


class DefaultAGNet(GeneralizedAGNet):
    def __init__(self,
                 backbone, n_outputs=2, task='classification',
                 inplanes=512, freeze_backbone=False, target_transform=None, class_weight=None):

        # helper func to get intermediate backbone outputs
        backbone = backbone
        mask_guider = MaskGuideNet(num_classes=n_outputs,
                                   inplanes=inplanes)
        # attentionmap_guider = AttentionMapGuideNet(use_cam=True)
        predictor = DefaultPredictor(
            n_inputs=inplanes, n_outputs=n_outputs, task=task, class_weight=class_weight, include_classifier=True)

        super(DefaultAGNet, self).__init__(backbone, mask_guider,
                                           predictor, freeze_backbone=freeze_backbone, target_transform=target_transform)


def default_agnet(opt, backbone, inplanes, n_outputs, load_pretrained_agnet=False, freeze_backbone=False, target_transform=None):
    net = DefaultAGNet(backbone, n_outputs=n_outputs, task=opt.task,
                       inplanes=inplanes, freeze_backbone=freeze_backbone, target_transform=target_transform, class_weight=opt.class_weight)

    if load_pretrained_agnet:
        load_pretrained_ckpt(net, opt.pretrained_path)
        freeze_layers(layers=net.children())

    return net
