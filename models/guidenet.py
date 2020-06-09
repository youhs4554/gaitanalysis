
import torch
from torch import nn
import torch.nn.functional as F
from .losses import MultiScaled_BCELoss, MultiScaled_DiceLoss, BinaryDiceLoss
from .utils import init_mask_layer, freeze_layers
from collections import OrderedDict

__all__ = [
    'GuideNet'
]


class GuideNet(nn.Module):
    def __init__(self, backbone, backbone_dims):

        super(GuideNet, self).__init__()

        del backbone.fc
        del backbone.avgpool

        feature_dim = 512   # use "C5" activation as a backbone feature

        # only compatible with backbones at `torchvision.models.video.*`
        self.backbone_layer = nn.Sequential(
            OrderedDict(dict(backbone.named_children())))

        self.mask_layer = nn.Sequential(
            nn.Conv3d(feature_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.perturb_layer = nn.Sequential(
            nn.Conv3d(feature_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.refined_mask_layer = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.path1 = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.path2 = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.criterion = BinaryDiceLoss()
        # self.criterion = nn.BCELoss()

    def forward(self, images, masks):
        # return (feats, guide_loss)
        x = self.backbone_layer(images)   # (N,512,2,7,7)

        # # upconv features
        # x = self.upconv(x)    # (N,512,16,112,112)

        detector_supervised_masks_pred = self.mask_layer(
            x)  # (N,1,16,112,112)

        eta = self.perturb_layer(x)

        # shake detector_supervised_masks by `eta`
        perturbed_masks_pred = self.refined_mask_layer(
            detector_supervised_masks_pred + eta)  # (N,1,16,112,112)

        l = self.path1(x)  # left
        r = self.path2(x)  # right

        out = perturbed_masks_pred * l + (1-perturbed_masks_pred) * r

        # matching_loss = - \
        #     torch.log(perturbed_masks_pred.tanh() *
        #               x.softmax(1).tanh()).mean()
        matching_loss = 1-(perturbed_masks_pred.tanh() *
                           x.softmax(1).tanh()).mean()

        # out = perturbed_masks_pred * x
        # # TODO. concat instead of multiplication...
        # out = torch.cat((x, perturbed_masks_pred), dim=1)
        # out = self.conv_1x1(out)

        if masks is None:
            return out, 0.0

        shapes = perturbed_masks_pred.size()[2:]
        masks = F.interpolate(masks, size=shapes, mode='nearest')

        # detector_guide_loss = self.criterion(
        #     detector_supervised_masks_pred, masks)

        # detector_guide_loss = (((detector_supervised_masks_pred > 0.5).float(
        # ) == masks).float() * detector_guide_loss).sum() / masks.sum()

        # # ((masks.tanh() * detector_supervised_masks_pred.tanh()) * detector_guide_loss)

        loss_dict = {
            'detector_guide_loss': self.criterion(
                detector_supervised_masks_pred, masks),
            'matching_loss': matching_loss
        }

        return out, masks, loss_dict
