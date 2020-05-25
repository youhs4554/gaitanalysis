import itertools
import itertools
from scipy.ndimage import zoom
import numpy as np
import torch


def get_mlist(guide=True):
    """
        moduel List to register to forward hook  ####
    """

    def get_mlist_without_guide():
        """
            moduel List to register to forward hook  ####
        """

        def bm_seq(prime, i): return [
            f'{prime}.{branch}_layers.{i}' for branch in ['backbone', 'matching']]

        m_list = [*list(itertools.chain(*[bm_seq('guidelessnet', i) for i in range(4)])),
                  'guidelessnet.conv_1x1',
                  'guidelessnet.fc']

        return m_list

    def get_mlist_with_guide():
        """
            moduel List to register to forward hook  ####
        """

        def bsm_seq(prime, i): return [f'{prime}.{branch}_layers.{i}' for branch in [
            'backbone', 'seg', 'matching']]

        m_list = [*list(itertools.chain(*[bsm_seq('agnet', i) for i in range(4)])),
                  'agnet.conv_1x1',
                  'agnet.fc']

        return m_list

    m_list = get_mlist_with_guide() if guide else get_mlist_without_guide()

    return m_list


def get_activation(m_dict, activation_layer, sample_duration, sample_size,
                   activation='relu'):
    A_k = m_dict[activation_layer]

    # move CUDA tensor to CPU
    A_k = A_k.cpu()

    # remove batch dim
    conv_out = A_k.data[0].numpy()

    # average
    conv_out = np.mean(conv_out, axis=0)

    # upsample grad_cam
    temporal_ratio = sample_duration / conv_out.shape[0]
    spatial_ratio = sample_size / conv_out.shape[1]

    conv_out = zoom(conv_out, (temporal_ratio,
                               spatial_ratio, spatial_ratio))

    if activation == 'relu':
        conv_out = np.maximum(conv_out, 0)
    elif activation == 'sigmoid':
        conv_out = 1/(1 + np.exp(-x))

    conv_out = conv_out / conv_out.max((1, 2))[:, None, None]

    return conv_out


def generate_grad_cam(m_dict, p_index, activation_layer, sample_duration, sample_size):
    y_c = torch.cat((m_dict['pretrained_agnet.fc'],
                     m_dict['agnet.fc']), 1)[0, p_index]
    A_k = m_dict[activation_layer]

    grad_val = torch.autograd.grad(y_c, A_k, retain_graph=True)[0].data

    # move CUDA tensor to CPU
    A_k = A_k.cpu()
    grad_val = grad_val.cpu()

    # remove batch dim
    conv_out = A_k.data[0].numpy()
    grad_val = grad_val[0].numpy()

    weights = np.mean(grad_val, axis=(1, 2, 3))
    #grad_cam = weights * conv_out

    grad_cam = np.zeros(dtype=np.float32, shape=conv_out.shape[1:])
    for k, w in enumerate(weights):
        grad_cam += w * conv_out[k]

    # upsample grad_cam
    temporal_ratio = sample_duration / grad_cam.shape[0]
    spatial_ratio = sample_size / grad_cam.shape[1]

    grad_cam = zoom(grad_cam, (temporal_ratio,
                               spatial_ratio, spatial_ratio))

    positive_grad_cam = np.maximum(grad_cam, 0)
    negative_grad_cam = np.maximum(-grad_cam, 0)

    positive_grad_cam = positive_grad_cam / \
        positive_grad_cam.max((1, 2))[:, None, None]
    negative_grad_cam = negative_grad_cam / \
        negative_grad_cam.max((1, 2))[:, None, None]

    return positive_grad_cam, negative_grad_cam


class ActivationMapProvider(object):
    def __init__(self, net, opt):
        self.net = net.module
        self.opt = opt
        self.enable_guide = self.opt.enable_guide

        self.register_forward_hook()

    def init_module_interface_obj(self):
        self.m_list = get_mlist(self.enable_guide)
        self.m_dict = {}

    def register_forward_hook(self):
        # before register hook, init interface_obj
        self.init_module_interface_obj()

        def hook(m, inp, out):
            layer_name = self.m_list.pop(0)
            self.m_dict[layer_name] = out

        # TODO. will be branched based on model_arch
        if self.enable_guide:
            # register hook for activation visualization
            self.net.fc.register_forward_hook(hook)
            self.net.agnet.conv_1x1.register_forward_hook(
                hook)
            for i in range(4):
                self.net.agnet.backbone[i].register_forward_hook(
                    hook)
                self.net.agnet.matching_layers[i].register_forward_hook(
                    hook)
                self.net.agnet.seg_layers[i].register_forward_hook(
                    hook)
        else:
            # register hook for activation visualization
            self.net.fc.register_forward_hook(hook)
            self.net.conv_1x1.register_forward_hook(hook)
            for i in range(4):
                self.net.backbone[i].register_forward_hook(hook)
                self.net.matching_layers[i].register_forward_hook(hook)

    def probe_forward_vals(self, img_tensor):
        img_tensor = img_tensor[None, :]

        with torch.no_grad():
            self.net.eval()
            _ = self.net(img_tensor)

    def compute(self, img_tensor, activation_layer=''):
        # feed img_tensor and intermediate activations are recorded into `m_dict`
        self.probe_forward_vals(img_tensor)

        # get activation results
        activation_result = get_activation(self.m_dict,
                                           sample_duration=self.opt.sample_duration,
                                           sample_size=self.opt.sample_size,
                                           activation_layer=activation_layer,
                                           activation='relu')

        # re-init interface_obj for later feeding
        self.init_module_interface_obj()

        return activation_result
