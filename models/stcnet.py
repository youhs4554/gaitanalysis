import torch
import torch.nn as nn
from .network_builder import Net
from .network import STCNet
from .default_predictor import DefaultPredictor
from .triplet_predictor import TripletPredictor
from .utils import freeze_layers, generate_backbone, load_pretrained_ckpt
import copy, os

__all__ = ["STC", "ConcatenatedSTC", "stcnet", "concatenated_stcnet"]


class STC(Net):
    def __init__(
        self,
        backbone,
        n_outputs=2,
        task="classification",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
        STC_squad="2,3,3",
    ):

        stc_network = STCNet(backbone, n_outputs=n_outputs, STC_squad=STC_squad)
        predictor = DefaultPredictor(
            n_inputs=backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(STC, self).__init__(
            stc_network, predictor, target_transform=target_transform
        )


class ConcatenatedSTC(Net):
    def __init__(
        self,
        backbone,
        pretrained_stcnet,
        n_outputs=2,
        task="regression",
        backbone_dims=[64, 64, 128, 256, 512],
        target_transform=None,
    ):
        assert task == "regression", "This module is for only regression task"

        stc_network = STCNet(backbone, n_outputs=n_outputs, STC_squad=STC_squad)
        predictor = DefaultPredictor(
            n_inputs=2 * backbone_dims[-1], n_outputs=n_outputs, task=task
        )

        super(ConcatenatedSTC, self).__init__(
            stc_network, predictor, target_transform=target_transform
        )

        # this pretrained STCNet should be frozen before init!
        self.pretrained_stcnet = pretrained_stcnet

    def forward(self, images, masks=None, targets=None, lambda_=None):
        mean_out, mean_feats, *_ = self.pretrained_stcnet(
            images,
            masks,
            targets=targets,
            lambda_=lambda_,
            return_intermediate_feats=False,
        )

        loss_dict = {}
        std_feats, STC_loss_dict, tb_dict = self.forward(
            images, masks, targets=targets, lambda_=lambda_
        )

        # merge with mean feature
        x = torch.cat([mean_feats, std_feats], 1)

        # model output
        out, predictor_loss_dict = self.predictor(x.mean((2, 3, 4)), targets)
        out = torch.cat([mean_out, out], 1)

        loss_dict.update(STC_loss_dict)
        loss_dict.update(predictor_loss_dict)

        if self.target_transform is not None:
            out = self.target_transform(out)

        return out, loss_dict, tb_dict


def stcnet(
    opt,
    backbone,
    backbone_dims,
    n_outputs,
    load_pretrained=False,
    target_transform=None,
):
    net = STC(
        backbone,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )

    if load_pretrained:
        if not opt.pretrained_path:
            raise ValueError("pretrained_path should be specified")
        if not os.path.exists(opt.pretrained_path):
            raise ValueError(f"pretrained_path `{opt.pretrained_path}` does not exist!")

        load_pretrained_ckpt(net, opt.pretrained_path)
        freeze_layers(layers=net.children())

    return net


def concatenated_stcnet(
    opt, backbone, backbone_dims, pretrained_stcnet, n_outputs, target_transform=None
):
    net = ConcatenatedSTC(
        backbone,
        pretrained_stcnet,
        n_outputs=n_outputs,
        task=opt.task,
        backbone_dims=backbone_dims,
        target_transform=target_transform,
    )
    return net
