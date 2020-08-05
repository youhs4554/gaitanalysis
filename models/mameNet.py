import torch
import torch.nn as nn
from .network_builder import Net
from .network import MameNet
from .default_predictor import DefaultPredictor
from .triplet_predictor import TripletPredictor
from .regional_predictor import RegionalPredictor
from .utils import freeze_layers, generate_backbone, load_pretrained_ckpt
import copy

__all__ = [
    "DefaultMAME",
    "RegionalMAME",
    "ConcatenatedMAME",
    "default_mameNet",
    "regional_mameNet",
    "concatenated_mameNet",
]


class DefaultMAME(Net):
    def __init__(
        self,
        backbone,
        n_outputs=2,
        task="classification",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
    ):

        mameNet = MameNet(backbone, n_class=n_outputs)
        predictor = DefaultPredictor(
            n_inputs=backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(DefaultMAME, self).__init__(
            mameNet, predictor, target_transform=target_transform
        )


class RegionalMAME(Net):
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

        mameNet = MameNet(backbone, backbone_dims, n_class=n_outputs)
        predictor = RegionalPredictor(
            n_inputs=backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(RegionalMAME, self).__init__(
            mameNet, predictor, target_transform=target_transform
        )


class ConcatenatedMAME(Net):
    def __init__(
        self,
        backbone,
        pretrained_mameNet,
        n_outputs=2,
        task="classification",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
    ):

        mameNet = MameNet(backbone, backbone_dims, n_class=n_outputs)
        predictor = DefaultPredictor(
            n_inputs=2 * backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(ConcatenatedMAME, self).__init__(
            mameNet, predictor, target_transform=target_transform
        )

        # this pretrained mameNet should be frozen before init!
        self.pretrained_mameNet = pretrained_mameNet

    def forward(self, images, masks=None, targets=None):

        mean_out, mean_feats, _ = self.pretrained_mameNet(
            images, masks, targets, return_intermediate_feats=True
        )

        loss_dict = {}
        std_feats, mame_loss_dict = self.forward_mameNet(images, masks)

        # merge with mean feature
        x = torch.cat([mean_feats, std_feats], 1)

        # model output
        out, predictor_loss_dict = self.predictor(x.mean((2, 3, 4)), targets)
        out = torch.cat([mean_out, out], 1)

        loss_dict.update(mame_loss_dict)
        loss_dict.update(predictor_loss_dict)

        if self.target_transform is not None:
            out = self.target_transform(out)

        return out, loss_dict


def default_mameNet(
    opt,
    backbone,
    backbone_dims,
    n_outputs,
    load_pretrained_mameNet=False,
    target_transform=None,
):
    net = DefaultMAME(
        backbone,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )

    if load_pretrained_mameNet:
        load_pretrained_ckpt(net, opt.pretrained_path)
        freeze_layers(layers=net.children())

    return net


def regional_mameNet(
    opt,
    backbone,
    backbone_dims,
    n_outputs,
    load_pretrained_mameNet=False,
    target_transform=None,
):
    net = RegionalMAME(
        backbone,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )

    if load_pretrained_mameNet:
        load_pretrained_ckpt(net, opt.pretrained_path)
        freeze_layers(layers=net.children())

    return net


def concatenated_mameNet(
    opt, backbone, backbone_dims, pretrained_mameNet, n_outputs, target_transform=None
):
    net = ConcatenatedMAME(
        backbone,
        pretrained_mameNet,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )
    return net
