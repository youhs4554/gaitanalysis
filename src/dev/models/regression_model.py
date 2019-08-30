from __future__ import print_function, division

from models.base_modules import *
from torch import nn
from torch.nn.functional import avg_pool2d, max_pool2d, softmax

def horizontal_pyramid_pooling(conv_out, n_groups, squeeze=True, reduce='spatial'):
    avg_out = []
    max_out = []

    for n_bins in [2 ** x for x in range(n_groups)]:
        bin_size = conv_out.size(2) // n_bins
        group = torch.split(conv_out, bin_size, dim=2)  # tuple of splitted arr

        pooling_kernel_size = [group[0].size(2), group[0].size(3)]

        if reduce=='height':
            pooling_kernel_size[1] = 1
        elif reduce=='width':
            pooling_kernel_size[0] = 1

        avg_stacked = torch.stack([avg_pool2d(x, kernel_size=pooling_kernel_size) for x in group])
        max_stacked = torch.stack([max_pool2d(x, kernel_size=pooling_kernel_size) for x in group])

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

        if backbone.module._get_name().lower()=='resnet':
            # (feats_len, feats_h, feats_w)
            _, self.fh, self.fw = backbone.module.avgpool.output_size

            # replace last layer with 1x1 conv
            backbone.module.fc = nn.Sequential(nn.Conv3d(backbone.module.fc.in_features, num_units, kernel_size=1),
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
    def __init__(self, n_groups, squeeze, reduce, cat_policy='group'):
        super(MultiScale_Pooling_Net, self).__init__()
        self.n_groups = n_groups
        self.squeeze = squeeze
        self.reduce = reduce
        self.cat_policy = cat_policy

    def forward(self, x):
        avg_out, max_out = horizontal_pyramid_pooling(x, self.n_groups, squeeze=self.squeeze, reduce=self.reduce)

        if self.cat_policy == 'group':
            # group-wise concat of group features
            avg_out = torch.cat(avg_out)  # (1+2+4, b, C)
            max_out = torch.cat(max_out)  # (1+2+4, b, C)
        elif self.cat_policy == 'channel':
            # channel-wise concat of group features
            avg_out = torch.cat([torch.cat(e.split(1), 2).squeeze(0) for e in avg_out], 1)  # (b, (1+2+4)*C, 1, W)
            max_out = torch.cat([torch.cat(e.split(1), 2).squeeze(0) for e in max_out], 1)  # (b, (1+2+4)*C, 1, W)

        x = avg_out + max_out

        return x


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

class MultiScale_1x1_base_Net(nn.Module):
    def __init__(self, input_dim, num_units, out_dim, n_groups=3):
        super(MultiScale_1x1_base_Net, self).__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.out_dim = out_dim
        self.n_groups = n_groups

        self.conv_1x1 = nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=1),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(True))

class MultiScale_1x1_Net(MultiScale_1x1_base_Net):
    def __init__(self, *args, **kwargs):
        super(MultiScale_1x1_Net, self).__init__(*args, **kwargs)

    def forward(self, *inputs):
        # x : (b, (1+2+4+...)*C, 1, W)
        x = inputs[0]

        # group merge type : 1x1 conv
        res = self.conv_1x1(x)  # (b, C, 1, W)

        # maxpool
        res = max_pool2d(res, kernel_size=(1, res.size(-1))).view(res.size(0), -1)  # (b,C)

        return res

class MultiScale_Attention_Net(nn.Module):
    def __init__(self, num_units, n_groups):
        super(MultiScale_Attention_Net, self).__init__()
        self.num_units = num_units
        self.n_groups = n_groups

    def forward(self, hpp_x, x):
        # hpp_x : (b,C*2**(n_groups-1),1,W)
        # x : (b,C,H,W)

        hpp_x = hpp_x.view(hpp_x.size(0), self.num_units, -1, hpp_x.size(-1))   # (b,C,2**(n_groups-1),W)

        # apply softmax along with groups index, to get importance of each grid group
        #hpp_x = softmax(hpp_x, dim=2)

        groups = torch.split(hpp_x, 1, dim=2)  # tuple of splitted arrs
        duplicated_groups = [ g.repeat(1,
                                       1,
                                       x.size(2) // 2**(self.n_groups-1),  # H/n_groups
                                       1) for g in groups ]

        alpha = torch.cat(duplicated_groups, dim=2)  # (b,C,H,W)

        # elementwise multiplication with attention value (alpha)
        res = x*alpha

        # identity connection ?
        #   res += x

        return res





class MultiScale_1x1_attention_Net(MultiScale_1x1_base_Net):
    def __init__(self, *args, **kwargs):
        super(MultiScale_1x1_attention_Net, self).__init__(*args, **kwargs)
        self.model = MultiScale_Attention_Net(num_units=self.num_units,
                                              n_groups=self.n_groups)

    def forward(self, hpp_x, x):
        # hpp_x : (b, (1+2+4+...)*C, 1, W)
        # x : (b,C,H,W)

        # group merge type : 1x1 conv
        hpp_x = self.conv_1x1(hpp_x)  # (b, C*n_groups, H, W)

        # attention mechanism
        res = self.model(hpp_x, x)

        # GMP
        res = max_pool2d(res, kernel_size=(res.size(2), res.size(3))).view(res.size(0), -1)  # (b,C))

        return res


class Naive_Flatten_Net(nn.Module):
    def __init__(self,
                 num_units,
                 n_factors,
                 backbone):

        super(Naive_Flatten_Net, self).__init__()

        # input_shape = (b,3,50,384,128)

        # feature embedding layer (common)
        self.backbone = BackboneEmbeddingNet(backbone, num_units)

        self.model = nn.Sequential(View(-1, num_units * self.backbone.fh * self.backbone.fw),
                                   nn.Linear(num_units * self.backbone.fh * self.backbone.fw, n_factors))


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

        self.multiscale_pooling = MultiScale_Pooling_Net(n_groups, squeeze=True, reduce='spatial', cat_policy='group')

        self.model = nn.Sequential(MultiScale_Addition_Net(input_dim=num_units, out_dim=num_units, n_groups=n_groups),
                                   nn.Dropout(drop_rate),
                                   nn.Linear(num_units, n_factors))

    def forward(self, x):

        x = self.backbone(x)
        x = self.multiscale_pooling(x)
        x = self.model(x)

        return x


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

        # feature embedding layer (common)
        self.backbone = BackboneEmbeddingNet(backbone, num_units)
        self.multiscale_pooling = MultiScale_Pooling_Net(n_groups, squeeze=False, reduce='height', cat_policy='channel')

        if attention:
            input_dim = num_units * sum([2 ** x for x in range(n_groups)])
            out_dim = num_units * (2 ** (n_groups - 1))

            multiscale_net = MultiScale_1x1_attention_Net(input_dim=input_dim, num_units=num_units,
                                                      out_dim=out_dim, n_groups=n_groups)
        else:
            input_dim = num_units * sum([2 ** x for x in range(n_groups)])
            out_dim = num_units

            multiscale_net = MultiScale_1x1_Net(input_dim=input_dim, num_units=num_units,
                                                      out_dim=out_dim, n_groups=n_groups)

        self.multiscale_net = multiscale_net

        self.regressor = nn.Sequential(nn.Dropout(drop_rate),
                                   nn.Linear(num_units, n_factors))


    def forward(self, x):
        feats = self.backbone(x)  # (b,C,H,W)
        hpp_x = self.multiscale_pooling(feats)  # (b, (1+2+4+...)*C, 1, W)
        res = self.multiscale_net(hpp_x, feats)
        res = self.regressor(res)

        return res