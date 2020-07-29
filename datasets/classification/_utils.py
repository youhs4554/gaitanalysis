import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tf_func
from PIL import Image, ImageDraw

__all__ = ["convert_box_coord", "generate_maskImg", "decompose_flow_frames"]


def convert_box_coord(x_center, y_center, box_width, box_height, W, H):
    # recover to original scale
    x_center = x_center * W
    box_width = box_width * W

    y_center = y_center * H
    box_height = box_height * H

    xmin = max(x_center - box_width / 2, 0)
    ymin = max(y_center - box_height / 2, 0)
    xmax = min(x_center + box_width / 2, W)
    ymax = min(y_center + box_height / 2, H)

    return (xmin, ymin, xmax, ymax)


def generate_maskImg(detection_res, query, W, H):
    x_center, y_center, box_width, box_height = detection_res.loc[query].values.T
    res = []
    if isinstance(x_center, np.ndarray):
        # multi-person
        for xc, yc, bw, bh in zip(x_center, y_center, box_width, box_height):
            res.append(convert_box_coord(xc, yc, bw, bh, W, H))
    else:
        # single-person
        res.append(convert_box_coord(x_center, y_center, box_width, box_height, W, H))

    # create mask image
    mask = Image.new("L", size=(W, H))
    for x in res:
        ImageDraw.Draw(mask).rectangle(x, fill="white")
    # to tensor
    mask = tf_func.to_tensor(mask).repeat(3, 1, 1)

    return mask

# decompose a flow_frame -> (flow_x, flow_y)
def decompose_flow_frames(flow_frames, transpose=True):
    res = []
    for t in range(flow_frames.size(1) - 1):
        x_comp, y_comp, _ = flow_frames[:,t].split(1, 0)
        x_comp = (x_comp.squeeze(0) - 0.5)/0.5
        y_comp = (y_comp.squeeze(0) - 0.5)/0.5

        flow_stack = np.stack((x_comp, y_comp))
        res.append(torch.FloatTensor(flow_stack))

    res = torch.stack(res, 0)
    if transpose:
        res = res.permute(1, 0, 2, 3)
    return res
