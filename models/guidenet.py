
import torch
from torch import nn
import torch.nn.functional as F
from .losses import MultiScaled_BCELoss, MultiScaled_DiceLoss, BinaryDiceLoss
from .utils import init_mask_layer, freeze_layers
from collections import OrderedDict, defaultdict

__all__ = [
    'GuideNet'
]


class GuideNet(nn.Module):
    def __init__(self, backbone, backbone_dims):
        super().__init__()

        del backbone.fc
        del backbone.avgpool

        # only compatible with backbone types implemented in `torchvision.models.video.*`
        self.backbone = nn.Sequential(
            OrderedDict(dict(backbone.named_children())))
        self.guide_module = GuideModule(self.backbone, backbone_dims[-1])

    def forward(self, images, masks):
        x, loss_dict = self.guide_module(images, masks)
        return x, loss_dict


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,  stride=stride,
                     groups=groups, dilation=dilation, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()

        self.conv1 = conv3x3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm3d(outplanes)
        self.bipass = nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=False),
            nn.BatchNorm3d(outplanes),
            nn.ReLU(inplace=True)
        ) if inplanes > outplanes else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # bi-pass
        identity = self.bipass(identity)

        x += identity

        x = self.relu(x)

        return x


class GuideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        x_out = x.clone()
        x_out[alpha < 0.5] = 0.0

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_x = grad_alpha = None

        grad_x = grad_output.clone()
        grad_x[alpha < 0.5] = 0.0

        return grad_x, grad_alpha


def init_weights(m, nonlinearity):
    def wrapper(m):
        if m.__class__.__name__.find("Conv") != -1:
            if nonlinearity == 'relu':
                torch.nn.init.kaiming_uniform_(
                    m.weight, mode='fan_in', nonlinearity='relu')
            elif nonlinearity == 'sigmoid':
                torch.nn.init.xavier_uniform_(m.weight)

    return wrapper


class GuideModule(nn.Module):
    def __init__(self, backbone_layer, feats_dim):

        super(GuideModule, self).__init__()

        self.backbone_layer = backbone_layer
        self.mask_layer = nn.Sequential(
            OrderedDict({
                "conv": ResidualBlock(feats_dim, feats_dim//4),
                "mask": conv1x1(feats_dim//4, 1)
            })
        )
        self.perturb_layer = nn.Sequential(
            OrderedDict({
                "conv": ResidualBlock(feats_dim, feats_dim//4),
                "mask": conv1x1(feats_dim//4, 1)
            })
        )
        self.embedding_layer = nn.Sequential(
            OrderedDict({
                "conv": ResidualBlock(1, feats_dim//4),
                "mask": conv1x1(feats_dim//4, feats_dim)
            })
        )

        for layer in [self.mask_layer.conv, self.perturb_layer.conv, self.embedding_layer]:
            layer.apply(init_weights(layer, nonlinearity='relu'))

        for layer in [self.mask_layer.mask, self.perturb_layer.mask]:
            layer.apply(init_weights(layer, nonlinearity='sigmoid'))

        self.guide_func = GuideFunction.apply
        self.criterion = nn.BCELoss()

    def forward(self, x, masks):
        loss_dict = defaultdict(float)

        # return (feats, guide_loss)
        x = self.backbone_layer(x)

        detector_supervised_masks_pred = self.mask_layer(
            x)
        detector_supervised_masks_pred = torch.sigmoid(
            detector_supervised_masks_pred)

        eta = self.perturb_layer(x)
        eta = torch.sigmoid(eta)

        # injects noise `eta` to `detector_supervised_masks`
        modified_masks_pred = detector_supervised_masks_pred + eta

        # embedding
        modified_masks_embed = self.embedding_layer(modified_masks_pred)
        modified_masks_embed = torch.relu(modified_masks_embed)

        # guide func which filters ROIS at both forward and backward path
        out = self.guide_func(
            x, modified_masks_pred.repeat((1, x.size(1), *[1]*(x.dim()-2))))

        # implemt. 1
        matching_loss = 1-(modified_masks_embed.tanh() * x.tanh()).mean()

        # implemt. 2
        # matching_loss = - \
        #     torch.log(1e-6+modified_masks_embed.tanh()*x.tanh()).mean()

        # implemt. 3 : distance between `modified_masks_embed` and `x` (backbone feature)
        # matching_loss = (modified_masks_embed - x.mean(0)).norm(2)

        loss_dict['matching_loss'] = matching_loss

        if masks is None:
            return out, loss_dict

        shapes = modified_masks_embed.size()[2:]
        masks = F.interpolate(masks, size=shapes, mode='nearest')

        mask_loss = self.criterion(
            detector_supervised_masks_pred, masks)
        loss_dict['mask_loss'] = mask_loss

        return out, loss_dict
