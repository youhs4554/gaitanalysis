import torch
import torch.nn as nn
from .generalized_agnet import GeneralizedAGNet
from .guidenet import GuideNet
from .default_predictor import DefaultPredictor
from .triplet_predictor import TripletPredictor
from .regional_predictor import RegionalPredictor
from .utils import freeze_layers, generate_backbone, load_pretrained_ckpt
import copy

__all__ = [
    "DefaultAGNet",
    "RegionalAGNet",
    "ConcatenatedAGNet",
    "default_agnet",
    "regional_agnet",
    "concatenated_agnet",
]


class DefaultAGNet(GeneralizedAGNet):
    def __init__(
        self,
        backbone,
        n_outputs=2,
        task="classification",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
    ):

        guider = GuideNet(backbone, n_class=n_outputs)
        predictor = DefaultPredictor(
            n_inputs=backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(DefaultAGNet, self).__init__(
            guider, predictor, target_transform=target_transform
        )


class RegionalAGNet(GeneralizedAGNet):
    def __init__(
        self,
        backbone,
        n_outputs=2,
        task="classification",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
    ):

        if task != "classification":
            raise ValueError("only classification is supported")

        guider = GuideNet(backbone, backbone_dims, n_class=n_outputs)
        predictor = RegionalPredictor(
            n_inputs=backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(RegionalAGNet, self).__init__(
            guider, predictor, target_transform=target_transform
        )


class ConcatenatedAGNet(GeneralizedAGNet):
    def __init__(
        self,
        backbone,
        pretrained_agnet,
        n_outputs=2,
        task="classification",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
    ):

        guider = GuideNet(backbone, backbone_dims, n_class=n_outputs)
        predictor = DefaultPredictor(
            n_inputs=2 * backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(ConcatenatedAGNet, self).__init__(
            guider, predictor, target_transform=target_transform
        )

        # this pretrained agnet should be frozen before init!
        self.pretrained_agnet = pretrained_agnet

    def forward(self, images, masks=None, targets=None):

        mean_out, mean_feats, _ = self.pretrained_agnet(
            images, masks, targets, return_intermediate_feats=True
        )

        loss_dict = {}
        std_feats, guide_loss_dict = self.forward_guider(images, masks)

        # merge with mean feature
        x = torch.cat([mean_feats, std_feats], 1)

        # model output
        out, predictor_loss_dict = self.predictor(x.mean((2, 3, 4)), targets)
        out = torch.cat([mean_out, out], 1)

        loss_dict.update(guide_loss_dict)
        loss_dict.update(predictor_loss_dict)

        if self.target_transform is not None:
            out = self.target_transform(out)

        return out, loss_dict


def default_agnet(
    opt,
    backbone,
    backbone_dims,
    n_outputs,
    load_pretrained_agnet=False,
    target_transform=None,
):
    net = DefaultAGNet(
        backbone,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )

    if load_pretrained_agnet:
        load_pretrained_ckpt(net, opt.pretrained_path)
        freeze_layers(layers=net.children())

    return net


def regional_agnet(
    opt,
    backbone,
    backbone_dims,
    n_outputs,
    load_pretrained_agnet=False,
    target_transform=None,
):
    net = RegionalAGNet(
        backbone,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )

    if load_pretrained_agnet:
        load_pretrained_ckpt(net, opt.pretrained_path)
        freeze_layers(layers=net.children())

    return net


def concatenated_agnet(
    opt, backbone, backbone_dims, pretrained_agnet, n_outputs, target_transform=None
):
    net = ConcatenatedAGNet(
        backbone,
        pretrained_agnet,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )
    return net
