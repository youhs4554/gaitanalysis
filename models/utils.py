import torch
import torch.nn as nn
import torchvision.models
from torch.nn import Parameter

from .i3res import I3ResNet
from .i3dense import I3DenseNet

__all__ = ["generate_backbone", "freeze_layers", "init_layers", "load_pretrained_ckpt"]


def load_pretrained_ckpt(net, pretrained_path=""):
    print(f"Load pretrained model from {pretrained_path}...")

    # laod pre-trained model
    pretrain = torch.load(pretrained_path, map_location=torch.device("cpu"))
    net.load_state_dict(pretrain["state_dict"])

    return net


def get_inflated_resnet(net_2d, net_3d):
    c_ix = 0
    bn_ix = 0
    net_2d_conv_layers = [m for m in list(net_2d.modules()) if isinstance(m, nn.Conv2d)]
    net_2d_bn_layers = [
        m for m in list(net_2d.modules()) if isinstance(m, nn.BatchNorm2d)
    ]

    for m in net_3d.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data = (
                net_2d_conv_layers[c_ix]
                .weight.data[:, :, None]
                .repeat((1, 1, m.kernel_size[0], 1, 1))
            )
            c_ix += 1
        if isinstance(m, nn.BatchNorm3d):
            m.weight.data = net_2d_bn_layers[bn_ix].weight.data
            bn_ix += 1

    return net_3d


def generate_backbone(type, pretrained=True):
    net = None
    if type in [
        "r2plus1d_34_32_kinetics",
        "r2plus1d_34_8_kinetics",
        "r2plus1d_34_32_ig65m",
        "r2plus1d_34_8_ig65m",
    ]:
        if type in ["r2plus1d_34_32_kinetics", "r2plus1d_34_8_kinetics"]:
            num_classes = 400
        elif type == "r2plus1d_34_32_ig65m":
            num_classes = 359
        elif type == "r2plus1d_34_8_ig65m":
            num_classes = 487

        net = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            type,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        dims = [64, 64, 128, 256, 512]

    if type in ["mc3_18", "r2plus1d_18", "r3d_18"]:
        net_init_func = getattr(torchvision.models.video, type)
        net = net_init_func(pretrained=pretrained)
        dims = [64, 64, 128, 256, 512]
    elif type.startswith("inflated"):
        resnet_arch = type.split("_")[1]
        resnet = getattr(torchvision.models, resnet_arch)(pretrained)
        net = I3ResNet(resnet)
        dims = [64, 256, 512, 1024, 2048]
        # from pprint import pprint
        # pprint("Inflated Net\n {}".format([ nn.Sequential(*list(net.children())[:i]).cuda()(torch.randn(1, 3, 16, 112, 112).cuda()).shape for i in range(7) ]))
        # pprint("Backbone Net\n {}".format([ nn.Sequential(*list(torchvision.models.video.r2plus1d_18().children())[:i]).cuda()(torch.randn(1, 3, 16, 112, 112).cuda()).shape for i in range(7) ]))

    # net will be uploaded to GPU later..
    if net is None:
        raise ValueError("invalid backbone Type")

    return net, dims


def freeze_layers(layers):
    for child in layers:
        for p in child.parameters():
            p.requires_grad = False


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def init_layers(layers):
    for child in layers:
        child.apply(init_weights)


def init_mask_layer(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if all(ks == 1 for ks in m.kernel_size):
            # zero weights for only 1x1 conv(last layer)
            m.weight.data.fill_(0.0)


def inflate_conv(
    conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False
):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride,
    )
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim, linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d, time_dim=1, time_padding=0, time_stride=None, time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode,
            )
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError(
                "{} is not among known pooling classes".format(type(pool2d))
            )

    return pool3d
