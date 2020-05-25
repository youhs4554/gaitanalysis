import torch
import torch.nn as nn
import torch.nn.functional as F
from .default_predictor import DefaultPredictor
from .utils import freeze_layers, generate_backbone


__all__ = [
    'FineTunedConvNet',
    'fine_tuned_convnet',
]


class FineTunedConvNet(nn.Module):
    def __init__(self, backbone, predictor, target_transform=None):
        super(FineTunedConvNet, self).__init__()

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.predictor = predictor
        self.target_transform = target_transform

    def forward(self, *inputs, targets=None):
        images, *_ = inputs  # ignore masks
        feats = self.backbone(images)
        out, predictor_loss_dict = self.predictor(feats, targets, None)

        if self.target_transform is not None:
            out = self.target_transform(out)

        return out, predictor_loss_dict


def fine_tuned_convnet(opt, backbone, backbone_dims, n_outputs, target_transform=None):
    predictor = DefaultPredictor(
        n_inputs=backbone_dims[-1], n_outputs=n_outputs, task=opt.task)
    baseline = FineTunedConvNet(
        backbone, predictor, target_transform=target_transform)

    return baseline
