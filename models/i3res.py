import math

import torch
from torch.nn import ReplicationPad3d
import torchvision
from . import utils


class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.conv_class = conv_class

        self.stem = torch.nn.Sequential(
            utils.inflate_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True),
            utils.inflate_batch_norm(resnet2d.bn1),
            torch.nn.ReLU(inplace=True),
            utils.inflate_pool(
                resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
            ),
        )

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = utils.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True,
            )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = utils.inflate_pool(resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = utils.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.conv_class:
            x = self.avgpool(x)
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x


def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = utils.inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = utils.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = utils.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = utils.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = utils.inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = utils.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        utils.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        utils.inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d


def inflated_resnet(arch, frame_nb=16):
    if arch == "resnet50":
        resnet = torchvision.models.resnet50(pretrained=True)
    elif arch == "resnet101":
        resnet = torchvision.models.resnet101(pretrained=True)
    elif arch == "resnet152":
        resnet = torchvision.models.resnet152(pretrained=True)

    inflated_resnetnet = I3ResNet(resnet, frame_nb=frame_nb)

    return inflated_resnetnet
