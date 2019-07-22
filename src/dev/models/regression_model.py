from __future__ import print_function, division
from models.base_modules import *

def horizetal_pyramid_pooling(conv_out, n_groups):
    gap_out = []
    gmp_out = []

    for n_bins in [2 ** x for x in range(n_groups)]:
        bin_size = conv_out.size(2) // n_bins
        group = torch.split(conv_out, bin_size, dim=2)  # tuple of splitted arr
        gap_out.append(torch.stack([x.mean((2,3)) for x in group]))
        gmp_out.append(torch.stack([x.max(-1)[0].max(-1)[0] for x in group]))

    gap_out = torch.cat(gap_out)  # (1+2+4, b, C)
    gmp_out = torch.cat(gmp_out)  # (1+2+4, b, C)

    res = gap_out + gmp_out
    res = res.permute(1,2,0)  # (b, C, 1+2+4)

    return res

class MultiScaleNet(nn.Module):
    def __init__(self, input_dim, out_dim, n_groups=3):
        super(MultiScaleNet, self).__init__()
        self.n_groups = n_groups

        for i in range(n_groups):
            self.add_module(f'conv_1x1_{i + 1}',
                            nn.Sequential(
                            nn.Conv1d(input_dim, out_dim, kernel_size=1),
                            nn.BatchNorm1d(out_dim),
                            nn.ReLU(True)))

    def forward(self, x):
        x = horizetal_pyramid_pooling(x, self.n_groups)  # (b, C, 1+2+4)

        s, n = (0, 1)  # for slicing

        res = []
        for name, layer in self.named_children():
            if name.startswith('conv_1x1'):
                _feats = x[:, :, s:s + n]  # (b,C,n)
                res.append(layer(_feats))  # (b,C/2,n)

                # update slicing info (s,n)
                s += n
                n *= 2

        res = torch.cat(res, 2).sum(2)  # sum along group index

        return res

class RegressionNet(nn.Module):
    def __init__(self,
                 num_units,
                 n_factors,
                 backbone,
                 drop_rate=None,
                 multi_scale=None,
                 n_groups=3):

        super(RegressionNet, self).__init__()

        # input_shape = (b,3,50,384,128)

        # replace last layer with 1x1 conv

        if backbone.module._get_name().lower()=='resnet':
            backbone.module.fc = nn.Sequential(nn.Conv3d(backbone.module.fc.in_features, num_units, kernel_size=1),
                                               nn.BatchNorm3d(num_units),
                                               nn.ReLU(True))
            # (feats_len, feats_h, feats_w)
            _, fh, fw = backbone.module.avgpool.output_size

        else:
            NotImplementedError('later..')

        # feature extraction layer (common)
        self.conv_net = nn.Sequential(*list(backbone.module.children()))

        if multi_scale:
            self.model = nn.Sequential(MultiScaleNet(input_dim=num_units, out_dim=num_units//2, n_groups=n_groups),
                                       nn.Dropout(drop_rate),
                                       nn.Linear(num_units//2, n_factors))
        else:
            self.model = nn.Sequential(View(-1, num_units*fh*fw),
                                       nn.Linear(num_units*fh*fw, n_factors))



    def forward(self, x):
        # simple avg along time axis
        x = self.conv_net(x).mean(2)
        x = self.model(x)

        return x