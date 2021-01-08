
from scipy.sparse.construct import identity
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCELoss
from .losses import MultiScaled_BCELoss, MultiScaled_DiceLoss, BinaryDiceLoss
from .utils import init_mask_layer, freeze_layers
from collections import OrderedDict

__all__ = [
    'GuideNet'
]


def conv_1x1x1(inplanes, planes):
    return nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)


def conv_3x3x3(inplanes, planes):
    return nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, bias=False)


class BottleneckConvBlock(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = conv_1x1x1(feature_dim, feature_dim//4)
        self.bn1 = nn.BatchNorm3d(feature_dim//4)
        self.conv2 = conv_3x3x3(feature_dim//4, feature_dim//4)
        self.bn2 = nn.BatchNorm3d(feature_dim//4)
        self.conv3 = conv_1x1x1(feature_dim//4, feature_dim)
        self.bn3 = nn.BatchNorm3d(feature_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class GuideNet(nn.Module):
    def __init__(self, backbone, backbone_dims):

        super(GuideNet, self).__init__()

        del backbone.fc
        del backbone.avgpool

        feature_dim = 512   # use "C5" activation as a backbone feature

        # only compatible with backbones at `torchvision.models.video.*`
        self.backbone_layer = nn.Sequential(
            OrderedDict(dict(backbone.named_children())))

        # mask prediction layer
        self.mask_layer = nn.Sequential(
            nn.Conv3d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # mask perturbation layer (applied on target masks)
        # with depth-wise separable convs
        self.perturb_layer = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim,
                      kernel_size=3, padding=1, groups=feature_dim),
            nn.ReLU(),
            nn.Conv3d(feature_dim, 1, kernel_size=1),
            nn.ReLU()
        )

        # dual-path ways which are weighted by refined masks
        self.path1 = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim,
                      kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.path2 = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim,
                      kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.criterion = BinaryDiceLoss()

    def forward(self, images, masks):
        # return (feats, guide_loss)
        feats = self.backbone_layer(images)   # (N,512,2,7,7)
        # # upconv features
        # x = self.upconv(x)    # (N,512,16,112,112)

        detector_supervised_masks_pred = self.mask_layer(
            feats)  # (N,1,16,112,112)

        l = self.path1(feats)  # left
        r = self.path2(feats)  # right

        out = detector_supervised_masks_pred * l + \
            (1-detector_supervised_masks_pred) * r

        if masks is None:
            return out, 0.0

        shapes = out.size()[2:]
        masks = F.interpolate(masks, size=shapes)

        # shake target masks
        eta = self.perturb_layer(feats)
        modified_masks = masks + eta
        loss_dict = {
            'detector_guide_loss': self.criterion(
                detector_supervised_masks_pred, modified_masks),
        }

        return out, modified_masks, loss_dict
