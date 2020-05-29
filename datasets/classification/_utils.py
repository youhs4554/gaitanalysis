import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "convert_box_coord",
    "generate_maskImg"
]


def convert_box_coord(x_center, y_center, box_width, box_height,
                      W, H):
    # recover to original scale
    x_center = x_center * W
    box_width = box_width * W

    y_center = y_center * H
    box_height = box_height * H

    xmin = max(x_center - box_width/2, 0)
    ymin = max(y_center - box_height/2, 0)
    xmax = min(x_center + box_width/2, W)
    ymax = min(y_center + box_height/2, H)

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
        res.append(convert_box_coord(
            x_center, y_center, box_width, box_height, W, H))

    # create mask image
    mask = Image.new('L', size=(W, H))
    for x in res:
        ImageDraw.Draw(mask).rectangle(x, fill="white")

    return mask
