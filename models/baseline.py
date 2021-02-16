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

        self.backbone = backbone
        self.predictor = predictor
        self.target_transform = target_transform

    def forward(self, *inputs, targets=None, enable_tsn=False):
        images, *_ = inputs  # ignore masks
        batch_size, nclips, *cdhw = images.size()

        if enable_tsn:
            images = images.view(-1, *cdhw)

        feats = self.backbone(images)

        # model output
        out, predictor_loss_dict = self.predictor(
            feats, targets, enable_tsn, batch_size)

        if self.target_transform is not None:
            out = self.target_transform(out)

        return out, predictor_loss_dict


def fine_tuned_convnet(opt, backbone, inplanes, n_outputs, target_transform=None):
    predictor = DefaultPredictor(
        n_inputs=inplanes, n_outputs=n_outputs, task=opt.task, class_weight=opt.class_weight)
    baseline = FineTunedConvNet(
        backbone, predictor, target_transform=target_transform)

    return baseline
