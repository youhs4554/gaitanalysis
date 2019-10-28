from __future__ import print_function, division

from models.base_modules import (
    View, MultiInputSequential, GAP, UpConv)
import torch
from torch import nn
from torch.nn.functional import (
    avg_pool2d, max_pool2d, softmax, relu, avg_pool3d, max_pool3d)
import math
import copy

pooling_dim = {
    'height': 2,
    'width': 3
}


def calc_PaddingNeed(input_size, kernel_size, stride=1, dilation=1, groups=1):
    effective_filter_size_rows = (kernel_size - 1) * dilation + 1
    out_size = (input_size + stride - 1) // stride
    padding_needed = max(
        0, (out_size - 1) * stride + effective_filter_size_rows - input_size)

    return padding_needed


def horizontal_pyramid_pooling(conv_out, n_groups,
                               squeeze=True, reduce='spatial'):
    avg_out = []
    max_out = []

    for n_bins in [2 ** x for x in range(n_groups)]:
        bin_size = conv_out.size(pooling_dim.get(reduce)) // n_bins

        group = torch.split(conv_out, bin_size, dim=pooling_dim.get(
            reduce))  # tuple of splitted arr

        pooling_kernel_size = [group[0].size(2), group[0].size(3)]

        if reduce == 'height':
            pooling_kernel_size[1] = 1
        elif reduce == 'width':
            pooling_kernel_size[0] = 1

        avg_stacked = torch.stack(
            [avg_pool2d(x, kernel_size=pooling_kernel_size) for x in group])
        max_stacked = torch.stack(
            [max_pool2d(x, kernel_size=pooling_kernel_size) for x in group])

        if squeeze:
            avg_stacked = avg_stacked.view(avg_stacked.size()[:-2])
            max_stacked = max_stacked.view(max_stacked.size()[:-2])

        avg_out.append(avg_stacked)
        max_out.append(max_stacked)

    return avg_out, max_out


class BackboneEmbeddingNet(nn.Module):
    def __init__(self,
                 backbone,
                 num_units):

        super(BackboneEmbeddingNet, self).__init__()

        # input_shape = (b,3,50,384,128)

        if backbone.module._get_name().lower() == 'resnet':
            # (feats_len, feats_h, feats_w)
            _, self.fh, self.fw = backbone.module.avgpool.output_size

            # replace last layer with 1x1 conv
            backbone.module.fc = nn.Sequential(
                nn.Conv3d(backbone.module.fc.in_features,
                          num_units, kernel_size=1),
                nn.BatchNorm3d(num_units),
                nn.ReLU(True))

        else:
            NotImplementedError('later..')

        # feature extraction layer (common)
        self.model = nn.Sequential(*list(backbone.module.children()))

    def forward(self, x):
        # input : (N,C,D,H,W)
        x = self.model(x)
        x = x.mean(2)

        return x


class MultiScale_Pooling_Net(nn.Module):
    def __init__(self, n_groups, squeeze, reduce,
                 cat_policy='group', get_input=False):
        super(MultiScale_Pooling_Net, self).__init__()
        self.n_groups = n_groups
        self.squeeze = squeeze
        self.reduce = reduce
        self.cat_policy = cat_policy
        self.get_input = get_input

    def forward(self, x):
        avg_out, max_out = horizontal_pyramid_pooling(
            x, self.n_groups, squeeze=self.squeeze, reduce=self.reduce)

        if self.cat_policy == 'group':
            # group-wise concat of group features
            avg_out = torch.cat(avg_out)  # (1+2+4, b, C)
            max_out = torch.cat(max_out)  # (1+2+4, b, C)
        elif self.cat_policy == 'channel':
            """
                if reduce=='height',
                    (b, (1+2+4)*C, 1, W)
                elif reduce=='width',
                    (b, (1+2+4)*C, H, 1)
            """
            # channel-wise concat of group features
            avg_out = torch.cat([torch.cat(e.split(1), 2).squeeze(0)
                                 for e in avg_out], 1)
            max_out = torch.cat([torch.cat(e.split(1), 2).squeeze(0)
                                 for e in max_out], 1)

        res = avg_out + max_out

        if self.get_input:
            return (res, x)

        return res


class MultiScale_Addition_Net(nn.Module):
    def __init__(self, input_dim, out_dim, n_groups=3):
        super(MultiScale_Addition_Net, self).__init__()
        self.n_groups = n_groups

        for i in range(n_groups):
            self.add_module(f'conv_1x1_{i + 1}',
                            nn.Sequential(
                                nn.Conv1d(input_dim, out_dim, kernel_size=1),
                                nn.ReLU(True)))

    def forward(self, x):
        x = x.permute(1, 2, 0)  # (b, C, 1+2+4)

        s, n = (0, 1)  # for slicing

        res = []
        for name, layer in self.named_children():
            if name.startswith('conv_1x1'):
                _feats = x[:, :, s:s + n]  # (b,C,n)
                res.append(layer(_feats))  # (b,C/2,n)

                # update slicing info (s,n)
                s += n
                n *= 2

        # group merge type : addition
        res = torch.cat(res, 2).sum(2)  # sum along group index

        return res


class MultiScale_Attention_Net(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super(MultiScale_Attention_Net, self).__init__()

        self.multiScale_feats_att = nn.Linear(embedding_dim, attention_dim)
        self.globalScale_feats_att = nn.Linear(embedding_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, emb_x, x):
        # emb_x : (b,embedding_dim)
        # x : (b,number_of_pixels,embedding_dim)

        att1 = self.multiScale_feats_att(emb_x)     # (b,attention_dim)

        # (b,number_of_pixels,attention_dim)
        att2 = self.globalScale_feats_att(x)

        att = self.full_att(relu(att1.unsqueeze(1)+att2)
                            ).squeeze(2)    # (b,number_of_pixels)
        alpha = softmax(att, dim=1)  # (b,number_of_pixels)

        attended_x = (x*alpha.unsqueeze(2)).sum(dim=1)     # (b,embedding_dim)

        return attended_x


class Naive_Flatten_Net(nn.Module):
    def __init__(self,
                 num_units,
                 n_factors,
                 backbone, drop_rate=0.0):

        super(Naive_Flatten_Net, self).__init__()

        # input_shape = (b,3,50,384,128)

        # feature embedding layer (common)
        self.backbone = BackboneEmbeddingNet(backbone, num_units)

        self.model = nn.Sequential(
            View(-1, num_units * self.backbone.fh * self.backbone.fw),
            nn.Dropout(drop_rate),
            nn.Linear(
                num_units * self.backbone.fh * self.backbone.fw, n_factors))

    def forward(self, x):

        x = self.backbone(x)
        x = self.model(x)

        return x


class HPP_Addition_Net(nn.Module):
    def __init__(self,
                 num_units,
                 n_factors,
                 backbone,
                 drop_rate=None,
                 n_groups=3):

        super(HPP_Addition_Net, self).__init__()

        # input_shape = (b,3,50,384,128)

        # feature embedding layer (common)
        self.backbone = BackboneEmbeddingNet(backbone, num_units)

        self.multiscale_pooling = MultiScale_Pooling_Net(
            n_groups, squeeze=True, reduce='spatial', cat_policy='group')

        self.model = nn.Sequential(
            MultiScale_Addition_Net(input_dim=num_units,
                                    out_dim=num_units, n_groups=n_groups),
            nn.Dropout(drop_rate),
            nn.Linear(num_units, n_factors))

    def forward(self, x):

        x = self.backbone(x)
        x = self.multiscale_pooling(x)
        x = self.model(x)

        return x


class Conv_1x1_Embedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv_1x1_Embedding, self).__init__()
        self.conv_1x1 = nn.Sequential(nn.Conv2d(*args, **kwargs),
                                      nn.BatchNorm2d(kwargs['out_channels']),
                                      nn.ReLU(True))

    def __call__(self, *inputs):
        if len(inputs) == 2:
            x_cat, x = inputs

            x_emb = self.conv_1x1(x_cat)    # (b,C,H,1) or (b,C,1,W)

            # avgpool
            x_emb = avg_pool2d(
                x_emb,
                kernel_size=(x_emb.size(2), x_emb.size(3))).view(
                    x_emb.size(0), -1)  # (b,C)

            # transpose x for attention
            x = x.permute(0, 2, 3, 1)   # (b,H,W,C)
            # (b,number_of_pixels,C)
            x = x.view(x.size(0), -1, x.size(3))

            return x_emb, x
        else:
            x_emb = self.conv_1x1(*inputs)

            # avgpool
            x_emb = avg_pool2d(
                x_emb,
                kernel_size=(x_emb.size(2), x_emb.size(3))).view(
                    x_emb.size(0), -1)  # (b,C)

            return x_emb


class HPP_1x1_Net(nn.Module):
    def __init__(self,
                 num_units,
                 n_factors,
                 backbone,
                 drop_rate=None,
                 attention=False,
                 n_groups=3):

        super(HPP_1x1_Net, self).__init__()

        # input_shape = (b,3,50,384,128)
        layers = []

        # feature embedding layer (common)
        layers.append(BackboneEmbeddingNet(backbone, num_units))

        # HPP layer : Multi-scaled feats
        layers.append(MultiScale_Pooling_Net(
            n_groups, squeeze=False, reduce='height',
            cat_policy='channel', get_input=attention))

        # conv_1x1 layer : Embed HPP feats
        layers.append(
            Conv_1x1_Embedding(
                in_channels=num_units * sum([2 ** x for x in range(n_groups)]),
                out_channels=num_units, kernel_size=1))

        if attention:
            # attention layer
            layers.append(MultiScale_Attention_Net(
                num_units, num_units//2))

        # regression layer : single FC
        layers.append(nn.Sequential(nn.Dropout(drop_rate),
                                    nn.Linear(num_units, n_factors)))

        self.model = MultiInputSequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class SpatialPyramid(nn.Module):
    def __init__(self, backbone, dilation_config,
                 num_units, n_factors, kernel_size=3, drop_rate=0.0):
        super(SpatialPyramid, self).__init__()

        # (feature_channels, feature_length, feature_height, feature_width)
        self._fC, self._fL, self._fH, self._fW =\
            backbone.module.fc.in_features, \
            *backbone.module.avgpool.output_size

        # except last fc layer from pretreaind backboneNet
        self.backbone = nn.Sequential(*list(backbone.module.children())[:-1])

        self.pyramid = nn.ModuleList(
            [self._makePyramid(
                _dilation=_dil, _num_units=num_units, _kernel_size=kernel_size)
                for _dil in dilation_config])

        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(num_units * len(dilation_config),
                      num_units, kernel_size=1),
            nn.BatchNorm3d(num_units),
            nn.ReLU()
        )

        self.reg = nn.Sequential(nn.Dropout(drop_rate),
                                 nn.Linear(num_units, n_factors))

    def _makePyramid(self, _dilation, _num_units, _kernel_size):

        layers = []

        padding_need = [math.floor(
            calc_PaddingNeed(
                input_size=_inSize,
                kernel_size=_kernel_size,
                dilation=_dilation)/2)
            for _inSize in [self._fL, self._fH, self._fW]]

        layers += [
            nn.Conv3d(self._fC, _num_units, _kernel_size,
                      padding=padding_need, dilation=_dilation),
            nn.BatchNorm3d(_num_units),
            nn.ReLU(),
            nn.Conv3d(
                _num_units, _num_units, kernel_size=1),
            nn.BatchNorm3d(_num_units),
            nn.ReLU(),
            nn.Conv3d(
                _num_units, _num_units, kernel_size=1)
        ]

        return nn.Sequential(*layers)

    def __call__(self, x):
        # x : (N,C,D,H,W)
        x = self.backbone(x)

        res = []
        for m in self.pyramid:
            res.append(m(x))

        # concat and conv_1x1
        x = torch.cat(res, dim=1)
        x = self.conv_1x1(x)
        x = torch.mean(x, (2, 3, 4))    # mean
        x = self.reg(x)                 # regression

        return x


class DeepFFT(nn.Module):
    def __init__(self, num_feats, n_factors, num_freq=1, drop_rate=0.0):
        super(DeepFFT,  self).__init__()
        self.num_feats = num_feats
        self.num_freq = num_freq

        self.convViz = self._build_coreNet(
            {
                'conv': {
                    'in_channels': [2048, 2048, 1024, 512, 512],
                    'out_channels': [2048, 1024, 512, 512, 256],
                    'kernel_size': [3]*4+[1],
                    'stride': [2]*4+[1],
                    'padding': [0]+[1]*3+[0]
                },
                'bn': {
                    'num_features': [2048, 1024, 512, 512, 256],
                }
            }
        )
        self.convFreq = self._build_coreNet(
            {
                'conv': {
                    'in_channels': [2048, 2048, 1024, 512, 512],
                    'out_channels': [2048, 1024, 512, 512, 256],
                    'kernel_size': [3]*4+[1],
                    'stride': [2]*4+[1],
                    'padding': [1]*4+[0]
                },
                'bn': {
                    'num_features': [2048, 1024, 512, 512, 256],
                }
            }
        )

        self.fc = nn.Sequential(nn.Dropout(drop_rate),
                                nn.Linear(256, n_factors))

    def _build_convBlock(self, *args, **kwargs):
        conv_args = args[0]
        bn_args = args[1]

        return nn.Sequential(nn.Conv1d(*conv_args),
                             nn.BatchNorm1d(*bn_args),
                             nn.ReLU())

    def _build_coreNet(self, layer_configs):
        layers = []
        conv_configs = layer_configs['conv']
        bn_configs = layer_configs['bn']

        for c, b in zip(zip(*conv_configs.values()),
                        zip(*bn_configs.values())):
            layers.append(self._build_convBlock(c, b))

        return nn.Sequential(*layers)

    def __call__(self, x):
        # x : (N,L,C)
        x = x.unsqueeze(-1)  # (N,L,C,1)
        x = x.permute(0, 2, 1, 3)  # (N, C, L, 1)

        im = torch.zeros_like(x)    # real component : (N,C,L,1)
        xr = x                      # imaginary component : (N,C,L,1)
        xc = torch.cat([xr, im], dim=3)

        # tensor with last dimension 2 (real+imag) , 1 is signal dimension
        fr = torch.fft(xc.cpu(), signal_ndim=1).to(xc.device)

        fft_r = fr[..., 0]
        fft_i = fr[..., 1]

        fft = torch.sqrt(fft_r**2+fft_i**2)

        # (N,hidden_size,num_freq)
        x_freq = self.convFreq(fft)
        x_freq = x_freq.mean(2)        # average => (N,hidden_size)
        x = self.fc(x_freq)

        return x


def freeze_layers(layers):
    for child in layers:
        for p in child.parameters():
            p.requires_grad = False


class ResUNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem = copy.deepcopy(backbone.stem)
        self.layer1 = copy.deepcopy(backbone.layer1)
        self.layer2 = copy.deepcopy(backbone.layer2)
        self.layer3 = copy.deepcopy(backbone.layer3)
        self.layer4 = copy.deepcopy(backbone.layer4)

        # freeze encoder layers!
        freeze_layers(layers=[self.stem, self.layer1, self.layer2])

        self.up_conv1 = UpConv(512+256, 256)
        self.up_conv2 = UpConv(256+128, 128)
        self.up_conv3 = UpConv(128+64, 64)

        self.out = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2),
                        mode='trilinear', align_corners=True),
            nn.Conv3d(64+64, 3, 1),
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.stem(x)    # (N,64,64,56,56)
        x2 = self.layer1(x1)    # (N,64,64,56,56)
        x3 = self.layer2(x2)     # (N,128,32,28,28)
        x4 = self.layer3(x3)     # (N,256,16,14,14)
        x5 = self.layer4(x4)     # (N,512,8,7,7)

        x = self.up_conv1(x5, x4)     # (N,256,16,14,14)
        x = self.up_conv2(x, x3)     # (N,128,32,28,28)
        x = self.up_conv3(x, x2)     # (N,64,64,56,56)

        x = torch.cat([x1, x], dim=1)   # (N,64+64,64,56,56)

        # segmentation output layer
        x = self.out(x)               # (N,3,64,112,112)

        return x


class SRegessionNet(nn.Module):
    def __init__(self, backbone, freeze=False):
        super(SRegessionNet, self).__init__()

        if freeze:
            freeze_layers(layers=backbone.children())

        self.backbone = backbone

        self.seg1 = nn.Sequential(
            nn.Conv3d(64, 2, kernel_size=1),
            nn.Sigmoid())

        self.seg2 = nn.Sequential(
            nn.Conv3d(64, 2, kernel_size=1),
            nn.Sigmoid())

        self.seg3 = nn.Sequential(
            nn.Conv3d(128, 2, kernel_size=1),
            nn.Sigmoid())

        self.seg4 = nn.Sequential(
            nn.Conv3d(256, 2, kernel_size=1),
            nn.Sigmoid())

        self.seg5 = nn.Sequential(
            nn.Conv3d(512, 2, kernel_size=1),
            nn.Sigmoid())

        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(64+128+256+512, 512, kernel_size=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True))

        self.reg = nn.Linear(512, 15)

    def forward(self, x):
        # x; (N,3,64,112,112)

        seg_list = []

        x = self.backbone.stem(x)    # (N,64,64,56,56)
        seg = self.seg1(x)
        x = x * seg[:, 0].unsqueeze(1)
        seg_list.append(seg)

        x = self.backbone.layer1(x)    # (N,64,64,56,56)
        seg = self.seg2(x)
        x = x * seg[:, 0].unsqueeze(1)
        x1 = max_pool3d(x, 2)       # (N,64,32,28,28)
        seg_list.append(seg)

        x = self.backbone.layer2(x)     # (N,128,32,28,28)
        x = x + x1
        seg = self.seg3(x)
        x = x * seg[:, 0].unsqueeze(1)
        x2 = max_pool3d(x, 4)       # (N,128,8,7,7)
        seg_list.append(seg)

        x = self.backbone.layer3(x)     # (N,256,16,14,14)
        seg = self.seg4(x)
        x = x * seg[:, 0].unsqueeze(1)  # (N,512,8,7,7)
        x3 = max_pool3d(x, 2)       # (N,256,8,7,7)
        seg_list.append(seg)

        x = self.backbone.layer4(x)     # (N,512,8,7,7)
        seg = self.seg5(x)
        x = x * seg[:, 0].unsqueeze(1)
        seg_list.append(seg)

        x = torch.cat([x1, x2, x3, x], dim=1)       # (N,64+128+256+512,8,7,7)
        x = self.conv_1x1(x)                        # (N,512,8,7,7)

        # spatiotemporal pooling
        x = avg_pool3d(x, (8, 7, 7)).view(-1, 512)

        # regression output
        x = self.reg(x)

        return x, seg_list
