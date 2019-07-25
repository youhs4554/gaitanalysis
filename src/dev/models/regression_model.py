from __future__ import print_function, division

from models.base_modules import *
from torch import nn
from torch.nn.functional import avg_pool2d, max_pool2d

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
            avg_stacked = avg_stacked.reshape(avg_stacked.size()[:-2])
            max_stacked = max_stacked.reshape(max_stacked.size()[:-2])

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

class MultiScale_Addition_Net(nn.Module):
    def __init__(self, input_dim, out_dim, n_groups=3):
        super(MultiScale_Addition_Net, self).__init__()
        self.n_groups = n_groups

        for i in range(n_groups):
            self.add_module(f'conv_1x1_{i + 1}',
                            nn.Sequential(
                            nn.Conv1d(input_dim, out_dim, kernel_size=1),
                            nn.BatchNorm1d(out_dim),
                            nn.ReLU(True)))

    def forward(self, x):
        avg_out, max_out = horizontal_pyramid_pooling(x, self.n_groups)

        avg_out = torch.cat(avg_out)  # (1+2+4, b, C)
        max_out = torch.cat(max_out)  # (1+2+4, b, C)

        x = avg_out + max_out

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


class MultiScale_1x1_Net(nn.Module):
    def __init__(self, input_dim, out_dim, n_groups=3):
        super(MultiScale_1x1_Net, self).__init__()
        self.n_groups = n_groups

        self.model = nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=1),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(True))


    def forward(self, x):
        avg_out, max_out = horizontal_pyramid_pooling(x, self.n_groups, squeeze=False, reduce='height')  # [ (1, b, C, 1, w), (2, b, C, 1, w), (4, b, C, 1, w), ...]

        # channel-wise concat of group features
        avg_out = torch.cat([torch.cat(e.split(1), 2).squeeze(0) for e in avg_out], 1)  # (b, (1+2+4)*C, 1, W)
        max_out = torch.cat([torch.cat(e.split(1), 2).squeeze(0) for e in max_out], 1)  # (b, (1+2+4)*C, 1, W)

        x = avg_out + max_out

        # group merge type : 1x1 conv
        res = self.model(x)  # (b, C, 1, W)

        # maxpool
        res = max_pool2d(res, kernel_size=(1,res.size(-1))).reshape(res.size(0), -1)  # (b,C)

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

        self.model = nn.Sequential(MultiScale_Addition_Net(input_dim=num_units, out_dim=num_units // 2, n_groups=n_groups),
                                   nn.Dropout(drop_rate),
                                   nn.Linear(num_units//2, n_factors))

    def forward(self, x):

        x = self.backbone(x)
        x = self.model(x)

        return x


class HPP_1x1_Net(nn.Module):
    def __init__(self,
                 num_units,
                 n_factors,
                 backbone,
                 drop_rate=None,
                 n_groups=3):

        super(HPP_1x1_Net, self).__init__()

        # input_shape = (b,3,50,384,128)

        # feature embedding layer (common)
        self.backbone = BackboneEmbeddingNet(backbone, num_units)

        self.model = nn.Sequential(MultiScale_1x1_Net(input_dim=num_units * sum([2 ** x for x in range(n_groups)]),
                                                      out_dim=num_units // 2, n_groups=n_groups),
                                   nn.Dropout(drop_rate),
                                   nn.Linear(num_units//2, n_factors))

    def forward(self, x):

        x = self.backbone(x)
        x = self.model(x)

        return x