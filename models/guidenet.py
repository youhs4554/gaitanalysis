
from functools import partial
from scipy.sparse.construct import identity
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.loss import BCELoss
from .losses import MultiScaled_BCELoss, MultiScaled_DiceLoss, BinaryDiceLoss
from .utils import init_mask_layer, freeze_layers
from collections import OrderedDict

__all__ = [
    'MaskGuideNet',
    'AttentionMapGuideNet'
]


def conv_1x1x1(inplanes, planes):
    return nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)


def conv_3x3x3(inplanes, planes):
    return nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, bias=False)


class BottleneckConvBlock(nn.Module):
    def __init__(self, inplanes=512):
        super().__init__()
        self.conv1 = conv_1x1x1(inplanes, inplanes//4)
        self.bn1 = nn.BatchNorm3d(inplanes//4)
        self.conv2 = conv_3x3x3(inplanes//4, inplanes//4)
        self.bn2 = nn.BatchNorm3d(inplanes//4)
        self.conv3 = conv_1x1x1(inplanes//4, inplanes)
        self.bn3 = nn.BatchNorm3d(inplanes)
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


def conv_depthwise(in_planes, out_planes):
    """depthwise convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1, groups=out_planes)


def conv_pointwise(in_planes, out_planes):
    """pointwise convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, padding=0)


class ConvHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, layers, dilation):
        d = OrderedDict()
        next_feature = in_channels
        for layer_ix, layer_features in enumerate(layers, 1):
            d["conv{}".format(layer_ix)] = nn.Conv3d(
                next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
            # d["bn{}".format(layer_ix)] = nn.BatchNorm3d(layer_features)
            d["relu{}".format(layer_ix)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d["predict"] = nn.Sequential(
            nn.ConvTranspose3d(
                next_feature, next_feature, kernel_size=(2, 4, 4), stride=(2, 4, 4)),
            nn.ReLU(True),
            # nn.Dropout3d(0.5),
            nn.Conv3d(next_feature, out_channels, 1, 1, 0)
        )

        super(ConvHead, self).__init__(d)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")

# class MaskGuideNet(nn.Module):
#     def __init__(self, num_classes, inplanes=512):

#         super(MaskGuideNet, self).__init__()

#         # mask prediction layer
#         self.mask_layer = nn.Sequential(
#             conv_1x1x1(inplanes, 1), nn.Sigmoid())

#         self.conv_dw = conv_depthwise(inplanes*2, inplanes)
#         self.conv_pw = conv_pointwise(inplanes, inplanes)

#         # mask perturbation layer (applied on target masks)
#         self.perturb_layer = conv_1x1x1(inplanes, 1)

#         self.relu = nn.ReLU()
#         self.criterion = BinaryDiceLoss()

#     def forward(self, x, masks):
#         residual = x

#         mask_pred = self.mask_layer(x)
#         mask_pred_repeated = mask_pred.repeat(1, x.size(1), 1, 1, 1)

#         # concat predicted mask with backbone feature maps in an alternate order
#         concat = torch.cat((x, mask_pred_repeated), dim=1)
#         concat = concat.view(x.size(0), 2, -1, *x.size()[2:]).transpose(1, 2)
#         concat = concat.reshape(x.size(0), -1, *x.size()[2:])

#         out = self.conv_dw(concat)
#         out = self.relu(out)
#         out = self.conv_pw(out)
#         out = self.relu(out)
#         # out += residual

#         if masks is None:
#             return out, 0.0

#         shapes = out.size()[2:]
#         masks = F.interpolate(masks, size=shapes)

#         # shake target masks
#         eta = self.perturb_layer(out)
#         modified_masks = masks + eta
#         loss_dict = {
#             'detector_guide_loss': self.criterion(
#                 mask_pred, modified_masks),
#         }

#         return out, modified_masks, loss_dict


# class MaskGuideNet(nn.Module):
#     def __init__(self, num_classes, inplanes=512):

#         super(MaskGuideNet, self).__init__()

#         self.num_classes = num_classes

#         # score frs
#         self.score_fr_c4 = nn.Conv3d(
#             inplanes//2, num_classes, kernel_size=1)
#         self.score_fr_c3 = nn.Conv3d(
#             inplanes//4, num_classes, kernel_size=1)

#         # upconvs
#         self.score2 = nn.ConvTranspose3d(
#             num_classes, num_classes, kernel_size=(1, 2, 2), stride=2)
#         self.score4 = nn.ConvTranspose3d(
#             num_classes, num_classes, kernel_size=(1, 2, 2), stride=2)

#         # prediction heads (simple)
#         self.mask_head = nn.ConvTranspose3d(
#             num_classes, 1, kernel_size=(2, 4, 4), stride=(2, 4, 4))
#         self.perturb_head = nn.ConvTranspose3d(
#             num_classes, 2, kernel_size=(2, 4, 4), stride=(2, 4, 4))
#         self.classScore_head = nn.ConvTranspose3d(
#             num_classes, num_classes, kernel_size=(2, 4, 4), stride=(2, 4, 4))

#         # # prediction heads (deep)
#         # self.mask_head = ConvHead(
#         #     num_classes, 1, layers=[128, 128], dilation=1)
#         # self.perturb_head = ConvHead(
#         #     num_classes, 2, layers=[128, 128], dilation=1)
#         # self.classScore_head = ConvHead(
#         #     num_classes, num_classes, layers=[128, 128], dilation=1)

#         midplanes = inplanes // 4

#         # for class inference
#         self.head = nn.Sequential(
#             nn.Conv3d(inplanes, midplanes, kernel_size=3,
#                       padding=1, bias=False),
#             nn.BatchNorm3d(midplanes),
#             nn.ReLU(True),
#             nn.Dropout3d(0.3),
#             nn.Conv3d(midplanes, num_classes, kernel_size=1),
#         )

#         self.bce_loss = nn.BCEWithLogitsLoss()

#         # reset conv parameters with kaiming_normal
#         for name, m in self.named_modules():
#             if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#                 if name in ["mask_head", "perturb_head"]:
#                     # all zero output at early stage of training for mask_logit and perturb_logit(i.e. gamma, beta)
#                     # note : sigmoid(0.0) -> 0.5
#                     nn.init.constant_(m.weight, 0.0)
#                     nn.init.constant_(m.bias, 0.0)
#                 else:
#                     nn.init.kaiming_normal_(
#                         m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, feats, gt_masks):
#         """
#             shapes
#             - c3 : (b,128,5,28,28)
#             - c4 : (b,256,3,14,14)
#             - c5 : (b,512,2,7,7)

#             spatially reduction ratio
#             - c3 : 1/4
#             - c4 : 1/8
#             - c5 : 1/16

#             temporally reduction ratio
#             - c3 : 1/2
#             - c4 : 1/4
#             - c5 : 1/8
#         """
#         c3, c4, c5 = feats

#         # upsample last conv feats
#         head_out = self.head(c5)  # (b,cls,2,7,7)
#         score2 = self.score2(head_out)  # 2x -> (b,cls,3,14,14)

#         skip_con1 = self.score_fr_c4(c4)  # (b,cls,3,14,14)
#         # Fuse1
#         Summed = skip_con1 + score2

#         score4 = self.score4(Summed)  # 4x -> (b,cls,5,28,28)

#         ###
#         skip_con2 = self.score_fr_c3(c3)  # (b,cls,5,28,28)
#         # Fuse2
#         Summed2 = skip_con2 + score4

#         # (b,cls,5,28,28) -> (b,1+2,10,112,112)
#         mask_logits = self.mask_head(Summed2)
#         perturb_factors = self.perturb_head(Summed2)
#         class_scores = self.classScore_head(Summed2)

#         # perturbation params
#         gamma, beta = torch.split(perturb_factors, 1, dim=1)

#         # modify gt_masks
#         modified_mask_logits = gt_masks * gamma + beta
#         modified_masks = torch.sigmoid(modified_mask_logits)

#         m1 = self.bce_loss(mask_logits, gt_masks)
#         m2 = self.bce_loss(mask_logits, modified_masks)  # aux

#         loss_dict = {
#             "mask_guide_loss": m1 + 0.5 * m2
#         }

#         # "head_out" : abstracted scores for classification
#         # "class_scores" : CAMs(Class Activation Maps) for guide mechanism

#         return head_out, modified_mask_logits, class_scores, loss_dict

def conv_depthwise(in_planes, out_planes):
    """depthwise convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1, groups=out_planes)


def conv_pointwise(in_planes, out_planes):
    """pointwise convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, padding=0)


class FCN(nn.Sequential):
    def __init__(self, in_channels, out_channels, layers, dilation):
        d = OrderedDict()
        next_feature = in_channels
        for layer_ix, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_ix)] = nn.Conv3d(
                next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_ix)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d["mask_fcn_logits"] = nn.Conv3d(next_feature, out_channels, 1, 1, 0)

        super(FCN, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu")


class MaskGuideNet(nn.Module):
    def __init__(self, num_classes, inplanes=512):

        super(MaskGuideNet, self).__init__()

        # mask prediction layer
        self.mask_layer = nn.Sequential(
            conv_1x1x1(inplanes, 1), nn.Sigmoid())
        # mask perturbation layer (applied on target masks)
        self.perturb_layer = conv_1x1x1(inplanes, 1)
        # refine layer
        self.refined_mask_layer = nn.Sequential(
            conv_1x1x1(1, 1), nn.Sigmoid())

        self.conv_dw = conv_depthwise(inplanes*2, inplanes)
        self.conv_pw = conv_pointwise(inplanes, inplanes)

        self.relu = nn.ReLU()
        self.criterion = BinaryDiceLoss()

    def forward(self, x, masks):
        residual = x

        mask_pred = self.mask_layer(x)
        eta = self.perturb_layer(x)

        # shake detector_supervised_masks by `eta`
        perturbed_masks_pred = self.refined_mask_layer(
            mask_pred + eta)  # (N,1,16,112,112)
        perturbed_masks_pred_repeated = perturbed_masks_pred.repeat(
            1, x.size(1), 1, 1, 1)

        # concat predicted mask with backbone feature maps in an alternate order
        concat = torch.cat((x, perturbed_masks_pred_repeated), dim=1)
        concat = concat.view(x.size(0), 2, -1, *x.size()[2:]).transpose(1, 2)
        concat = concat.reshape(x.size(0), -1, *x.size()[2:])

        out = self.conv_dw(concat)
        out = self.relu(out)
        out = self.conv_pw(out)
        out = self.relu(out)
        # out += residual

        if masks is None:
            return out, 0.0

        shapes = out.size()[2:]
        masks = F.interpolate(masks, size=shapes)

        loss_dict = {
            'detector_guide_loss': self.criterion(
                mask_pred, masks),
        }

        return out, perturbed_masks_pred, loss_dict


# class MaskGuideNet(nn.Module):
#     def __init__(self, num_classes, inplanes=512):

#         super(MaskGuideNet, self).__init__()

#         hidden = inplanes

#         # FCNs
#         self.mask_predictor = self.make_mask_predictor(
#             inplanes=inplanes, hidden=hidden, n_heads=2)
#         self.pfpl = self.make_pfpl(inplanes=hidden, hidden=hidden, n_heads=2)

#         self.separable_conv = nn.Sequential(
#             conv_depthwise(inplanes*2, hidden),
#             nn.BatchNorm3d(hidden),
#             nn.ReLU(True),
#             conv_pointwise(hidden, hidden),
#             nn.BatchNorm3d(hidden),
#             nn.ReLU(True)
#         )

#         # for class inference
#         self.fc6 = nn.Sequential(
#             nn.Conv3d(hidden, hidden, kernel_size=3,
#                       padding=1, dilation=1),  # fc6
#             nn.ReLU(True)
#         )
#         self.fc7_0 = nn.Sequential(
#             nn.Conv3d(hidden, hidden, kernel_size=3,
#                       padding=1, dilation=1),  # fc6
#             nn.ReLU(True),
#         )
#         self.classifier_0 = nn.Conv3d(
#             hidden, num_classes, kernel_size=1, dilation=1)

#         self.bce_loss = nn.BCEWithLogitsLoss()

#         # reset conv parameters with kaiming_normal
#         for m in self.modules():
#             if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')

#     @staticmethod
#     def build_FCN(inplanes, layers, out_channels=1, mask_dilation=1):
#         return FCN(inplanes, out_channels, layers, mask_dilation)

#     def make_mask_predictor(self, inplanes, hidden=None, n_heads=4):
#         ''' Mask prediction layer '''
#         if hidden is None:
#             hidden = 256
#         layers = [hidden] * n_heads
#         out_channels = 1  # single channel output
#         mask_dilation = 1

#         return self.build_FCN(inplanes, layers, out_channels, mask_dilation)

#     def make_pfpl(self, inplanes, hidden=None, n_heads=4):
#         ''' Perturb factor prediction layer(PFPL) '''
#         if hidden is None:
#             hidden = 256
#         layers = [hidden] * n_heads
#         out_channels = 2  # dual channel output each channel represents "gamma", "beta"
#         mask_dilation = 1

#         return self.build_FCN(inplanes, layers, out_channels, mask_dilation)

#     def get_modified_mask_logits(self):
#         return self.modified_mask_logits

#     def modify_gt_mask_logits(self, x, gt_masks_down):
#         """
#             modify down-sampled gt_masks with FCN named as perturb factor prediction layer(pfpl)
#         """

#         # (b,2,d,h,w)
#         perturb_factors = self.pfpl(x)

#         # upsample
#         gamma, beta = torch.split(perturb_factors, 1, dim=1)

#         # modify gt_masks_down
#         modified_mask_logits = gt_masks_down * gamma + beta

#         return modified_mask_logits

#     def output_inference(self, x):
#         # x = F.dropout(x, 0.5, training=self.training)
#         x = self.fc6(x)
#         # x = F.dropout(x, 0.5, training=self.training)
#         x = self.fc7_0(x)
#         # x = F.dropout(x, 0.5, training=self.training)
#         out = self.classifier_0(x)
#         return out, x

#     def masked_depthwise_separable_conv(self, x, mask_logits):
#         mask_probs = torch.sigmoid(mask_logits)
#         mask_probs_repeated = mask_probs.repeat(1, x.size(1), 1, 1, 1)

#         # concat predicted mask with backbone feature maps in an alternate order
#         concat = torch.cat((x, mask_probs_repeated), dim=1)
#         concat = concat.view(x.size(0), 2, -1, *
#                              x.size()[2:]).transpose(1, 2)
#         concat = concat.reshape(x.size(0), -1, *x.size()[2:])

#         # depthwise_separable_convs
#         out = self.separable_conv(concat)

#         # # skip connect with mask-out features
#         # out = out + (x * mask_logits.ge(0.5).float()) + x

#         return out

#     def forward(self, x, gt_masks):

#         feats = x

#         mask_logits = self.mask_predictor(feats)

#         # down-sample gt_masks
#         gt_masks_down = F.interpolate(
#             gt_masks, size=mask_logits.size()[-3:], mode="nearest")

#         out = self.masked_depthwise_separable_conv(feats, mask_logits)
#         out, last_feats = self.output_inference(out)
#         ''' options
#             - 1 : x (n,256,3,14,14)
#             - 2 : last_feats (n,256,3,14,14)
#             - 3 : feats (n,256,3,14,14)
#         '''

#         modified_mask_logits = self.modify_gt_mask_logits(
#             feats, gt_masks_down)
#         self.modified_mask_logits = modified_mask_logits

#         modified_masks = torch.sigmoid(modified_mask_logits)

#         m1 = self.bce_loss(mask_logits, gt_masks_down)
#         m2 = self.bce_loss(mask_logits, modified_masks)  # aux

#         loss_dict = {
#             "mask_guide_loss": m1 + 0.5 * m2
#         }

#         return out, modified_mask_logits, loss_dict


class AttentionMapGuideNet(nn.Module):
    def __init__(self, use_cam=True) -> None:
        super().__init__()
        self.use_cam = use_cam
        self.bce_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def normalize_attnmap(attn_map):
        attn_shape = attn_map.size()

        batch_mins, _ = torch.min(attn_map.view(
            attn_shape[0:-3] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(attn_map.view(
            attn_shape[0:-3] + (-1,)), dim=-1, keepdim=True)
        attn_map_norm = torch.div(attn_map.view(attn_shape[0:-3] + (-1,))-batch_mins,
                                  batch_maxs - batch_mins)
        attn_map_norm = attn_map_norm.view(attn_shape)

        return attn_map_norm

    def return_CAM(self, feature_maps, weight_softmax, class_idx, size_upsample=(10, 112, 112), normalize=True):
        # generate the class -activation maps and upsample to 10x112x112
        bz, nc, d, h, w = feature_maps.shape
        output_cam = []
        for batch_idx, idx in enumerate(class_idx):
            beforeDot = feature_maps[batch_idx].view(nc, d*h*w)
            cam = torch.matmul(weight_softmax[idx].data, beforeDot)
            if size_upsample is not None:
                cam = cam.view(1, 1, d, h, w)
                cam = F.interpolate(cam, size_upsample,
                                    mode="trilinear", align_corners=False).squeeze()
            else:
                cam = cam.view(d, h, w)

            if normalize:
                cam = self.normalize_attnmap(cam)
            output_cam.append(cam)

        output_cam = torch.stack(output_cam, 0).unsqueeze(1)  # (b,1,d,h,w)
        return output_cam

    def return_attn_map(self, feature_maps, class_idx, normalize=True):
        class_idx = class_idx.long()

        # feature_maps : (b,nc,d,h,w)
        feature_sizes = feature_maps.size()
        attn_map = torch.zeros(
            feature_sizes[0], *feature_sizes[2:])  # (b,d,h,w)
        attn_map = Variable(attn_map.to(feature_maps.device))

        for batch_ix, fm in enumerate(feature_maps):
            attn_map[batch_ix] = fm[class_idx.data[batch_ix]]

        if normalize:
            attn_map = self.normalize_attnmap(attn_map)

        attn_map = attn_map.unsqueeze(1)  # (b,1,d,h,w)

        return attn_map

    def forward(self, feature_maps, modified_mask_logits,
                logit, weight_softmax=None, gt_labels=None):

        # if self.training:
        #     # use gt_labels during training
        #     idxs = gt_labels.long()
        # else:
        #     # use predicted labels during evaluation
        #     probs = F.softmax(logit, dim=1).data
        #     idxs = probs.argmax(1)

        idxs = gt_labels.long()

        size_upsample = modified_mask_logits.size()[-3:]

        if self.use_cam:
            attn_maps = self.return_CAM(feature_maps, weight_softmax,
                                        class_idx=idxs, size_upsample=size_upsample, normalize=True)
        else:
            feature_maps_up = F.upsample(
                feature_maps, size_upsample, mode="trilinear", align_corners=False)
            attn_maps = self.return_attn_map(
                feature_maps_up, class_idx=idxs, normalize=True)

        loss_dict = {
            'attention_guide_loss': self.bce_loss(modified_mask_logits, attn_maps.detach())
        }

        return attn_maps, loss_dict
