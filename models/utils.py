import torch
import torch.nn as nn
import torchvision.models

__all__ = [
    'generate_backbone',
    'freeze_layers',
    'init_layers',
    'load_pretrained_ckpt',
]


def load_pretrained_ckpt(net, pretrained_path=''):
    print(f"Load pretrained model from {pretrained_path}...")

    # laod pre-trained model
    pretrain = torch.load(pretrained_path,
                          map_location=torch.device('cpu'))
    net.load_state_dict(pretrain['state_dict'])

    return net


def get_inflated_resnet(net_2d, net_3d):
    c_ix = 0
    bn_ix = 0
    net_2d_conv_layers = [m for m in list(
        net_2d.modules()) if isinstance(m, nn.Conv2d)]
    net_2d_bn_layers = [m for m in list(net_2d.modules())
                        if isinstance(m, nn.BatchNorm2d)]

    for m in net_3d.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data = net_2d_conv_layers[c_ix].weight.data[:, :, None].repeat(
                (1, 1, m.kernel_size[0], 1, 1))
            c_ix += 1
        if isinstance(m, nn.BatchNorm3d):
            m.weight.data = net_2d_bn_layers[bn_ix].weight.data
            bn_ix += 1

    return net_3d


def generate_backbone(type, pretrained=True):
    net = None
    if type in ['r2plus1d_34_32_kinetics', 'r2plus1d_34_8_kinetics', 'r2plus1d_34_32_ig65m', 'r2plus1d_34_8_ig65m']:
        if type in ['r2plus1d_34_32_kinetics', 'r2plus1d_34_8_kinetics']:
            num_classes = 400
        elif type == 'r2plus1d_34_32_ig65m':
            num_classes = 359
        elif type == 'r2plus1d_34_8_ig65m':
            num_classes = 487

        net = torch.hub.load("moabitcoin/ig65m-pytorch",
                             type, num_classes=num_classes, pretrained=pretrained)
        dims = [64, 64, 128, 256, 512]

    if type in ['mc3_18', 'r2plus1d_18', 'r3d_18']:
        net_init_func = getattr(torchvision.models.video, type)
        net = net_init_func(pretrained=pretrained)
        dims = [64, 64, 128, 256, 512]
    elif type == 'inflated':
        resnet_2d = torchvision.models.resnet18(pretrained)
        resnet_3d = torchvision.models.video.r3d_18(pretrained)
        # inflate 2d weights to
        net = get_inflated_resnet(net_2d=resnet_2d, net_3d=resnet_3d)
        dims = [64, 64, 128, 256, 512]

    # net will be uploaded to GPU later..
    if net is None:
        raise ValueError('invalid backbone Type')

    return net, dims


def freeze_layers(layers):
    for child in layers:
        for p in child.parameters():
            p.requires_grad = False


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def init_layers(layers):
    for child in layers:
        child.apply(init_weights)


def init_mask_layer(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if all(ks == 1 for ks in m.kernel_size):
            # zero weights for only 1x1 conv(last layer)
            m.weight.data.fill_(0.0)
