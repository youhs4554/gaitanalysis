import torch.nn as nn
import torchvision.transforms.functional as tf_func
import random
import math
import numbers
import collections
import copy
import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Dict

try:
    import accimage
except ImportError:
    accimage = None


class VideoAugmentator:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:

        # seed is given at the beginning
        seed = random.randint(-1e8, 1e8)

        frames = kwargs.get("frames", None)
        assert isinstance(
            frames, list
        ), "frames should be List any (torch | numpy) arrays not {}".format(
            type(frames)
        )
        assert len(frames) > 0, "Empty frames is not allowed!"

        mask_frames = kwargs.get("mask_frames", [None] * len(frames))

        # ensure same length of mask_frames
        for k, v in [("mask", mask_frames)]:
            assert isinstance(
                v, list
            ), "{}_frames should be List any (torch | numpy) arrays not {}".format(
                k, type(v)
            )
            assert len(frames) == len(
                v
            ), "Length of frames and {}_frames should be identical".format(k)

        res = collections.defaultdict(list)
        for t in range(len(frames)):
            random.seed(seed)
            img = frames[t]
            mask = mask_frames[t]

            img, mask = [
                x.permute(1, 2, 0).mul(255).byte()
                if isinstance(x, torch.FloatTensor)
                else x
                for x in [img, mask]
            ]

            mask = mask[..., 0]  # use single-channel mask

            img_arr, mask_arr = [
                np.array(x) if isinstance(x, (torch.Tensor, np.ndarray)) else None
                for x in [img, mask]
            ]
            aug = self.transform(image=img_arr, mask=mask_arr)
            for key in aug:
                res[key + "_frames"].append(aug[key])

        for key in res:
            res[key] = torch.stack(res[key]).permute(0, 2, 3, 1)

        return res


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self, vid):
        for t in self.transforms:
            if hasattr(t, "randomize_parameters"):
                if t.__class__.__name__ in ["RandomResizedCrop3D", "RandomCrop3D"]:
                    t.randomize_parameters(vid[0])
                else:
                    t.randomize_parameters()

            vid = t(vid)


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (
            isinstance(size, collections.Iterable) and len(size) == 2
        )
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class CornerCrop(object):
    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ["c", "tl", "tr", "bl", "br"]

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == "c":
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.0))
            y1 = int(round((image_height - th) / 2.0))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == "tl":
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == "tr":
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == "bl":
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == "br":
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[
                random.randint(0, len(self.crop_positions) - 1)
            ]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        scales,
        size,
        interpolation=Image.BILINEAR,
        crop_positions=["c", "tc", "bc", "tl", "tr", "bl", "br"],
    ):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        center_x = image_width // 2
        center_y = image_height // 2
        box_half = crop_size // 2

        if self.crop_position == "c":
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == "tc":
            x1 = center_x - box_half
            y1 = 0
            x2 = center_x + box_half
            y2 = crop_size
        elif self.crop_position == "bc":
            x1 = center_x - box_half
            y1 = image_height - crop_size
            x2 = center_x + box_half
            y2 = image_height
        elif self.crop_position == "tl":
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == "tr":
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == "bl":
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == "br":
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[
            random.randint(0, len(self.crop_positions) - 1)
        ]


class MultiScaleRandomCrop(object):
    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()


class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class LoopTemporalCrop(object):
    def __init__(self, step_between_clips=16):
        self.step_between_clips = step_between_clips
        self.pad_method = LoopPadding(size=step_between_clips)

    def __call__(self, vid):
        # vid : list of PIL images
        np_vid = np.stack([np.array(pic) for pic in vid])
        tensor_vid = torch.from_numpy(np_vid)  # (T,H,W,C)

        n_frames = len(tensor_vid)

        out = []
        for i in range(0, n_frames, self.step_between_clips):
            clip_pts = list(range(i, min(n_frames, i + self.step_between_clips)))
            # loop-padding to guarantee same length
            clip_pts = self.pad_method(clip_pts)
            clip = [Image.fromarray(t.numpy()) for t in tensor_vid[clip_pts]]

            out.append(clip)  # list of PIL images

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[: self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalSegmentsSampling(object):
    """TSN sampling method
    ; Samples random indices from segments, which the duration of each segment is equal.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, n_seg=16):
        self.n_seg = n_seg

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        if len(frame_indices) < self.n_seg:
            # duplicated(allow duplicated elemnts) sampling (random)
            frame_indices = RandomDuplicatedSampling(self.n_seg)(frame_indices)

        seg_counts = collections.defaultdict(int)
        finish = False
        while not finish:
            for i in range(self.n_seg):
                seg_counts[i] += 1
                if sum(seg_counts.values()) == len(frame_indices):
                    finish = True
                    break

        seg_counts = collections.OrderedDict(
            {k: v for k, v in sorted(seg_counts.items(), key=lambda x: x[0])}
        )

        res = []

        for k, v in seg_counts.items():
            current_seg = frame_indices[:v]

            # random sample from current segment
            rand_sample = random.choice(current_seg)
            res.append(rand_sample)

            # update frame_indices (pop operation by replacing original list...)
            frame_indices = frame_indices[v:]

        return res


class TemporalSlidingWindow(object):
    """Temporally sliding windows for a given frame_indices

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        from torchvision.datasets.video_utils import unfold

        if len(frame_indices) < self.size:
            # duplicated sampling (non-random)
            # if frame_indices is given as list(range(13)), and size as 64
            # results : [[0] * 5, [1] * 5, ... [12]*4] ; last element is residual
            frame_indices = DuplicatedSampling(self.size)(frame_indices)

        windows = unfold(torch.as_tensor(frame_indices), self.size, self.size)
        res = []
        for win in windows:
            res.append(win)
        return res


class TemporalUniformSampling(object):
    """Temporally sample the given frame indices with uniformly splited chunks.

    Args:
        n_chunks (int): Desired number of chunks to split.
        size (int): Desired number of frames per each chunk (same as sample_duration)
    """

    def __init__(self, n_chunks, size=16):
        self.n_chunks = n_chunks
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: nested list of frame indices for each chunk.
        """
        chunk_counts = collections.defaultdict(int)

        finish = False
        while not finish:
            for i in range(self.n_chunks):
                chunk_counts[i] += 1
                if sum(chunk_counts.values()) == len(frame_indices):
                    finish = True
                    break

        res = []

        chunk_counts = collections.OrderedDict(
            {k: v for k, v in sorted(chunk_counts.items(), key=lambda x: x[0])}
        )
        for ix, (k, v) in enumerate(chunk_counts.items()):
            current_chunk = frame_indices[:v]
            if len(current_chunk) > self.size:
                # # uniform sampling for current_chunk
                # current_chunk = np.array(current_chunk)[
                #     np.linspace(
                #         0, len(current_chunk), num=self.size, endpoint=False, dtype=int
                #     )
                # ]
                current_chunk = np.array(current_chunk)[
                    TemporalCenterCrop(size=self.size)(list(range(len(current_chunk))))
                ]
            else:
                # loop padding for frame_indices shorter than the `sample_duration`
                current_chunk = np.array(current_chunk)[
                    LoopPadding(self.size)(list(range(len(current_chunk))))
                ]

            # update frame_indices (pop operation by replacing original list...)
            frame_indices = frame_indices[v:]
            res.append(current_chunk)
        return res


class DuplicatedSampling(object):
    def __init__(self, size):
        self.size = size

    def compute_duplications(self, VectorSize, Sum):
        c = int(Sum / VectorSize)
        smallest_c = copy.copy(c)
        while True:
            residual = Sum - c * (VectorSize - 1)
            if residual - (VectorSize - 1) < smallest_c:
                break
            c += 1
        x = [c for _ in range(1, VectorSize)]
        x.append(Sum - sum(x))
        return x

    def __call__(self, frame_indices):
        # duplicated indices (not random)
        dups = self.compute_duplications(len(frame_indices), self.size)
        res = []
        for ix, e in enumerate(frame_indices):
            n = dups[ix]
            res.extend([e] * n)
        return res


class RandomDuplicatedSampling(object):
    def __init__(self, size):
        self.size = size

    def random_compute_duplications(self, VectorSize, Sum):
        x = [random.randint(0, Sum) for _ in range(1, VectorSize)]
        x.extend([0, Sum])
        x.sort()
        return [x[i + 1] - x[i] for i in range(VectorSize)]

    def __call__(self, frame_indices):
        dups = self.random_compute_duplications(len(frame_indices), self.size)
        res = []
        for ix, e in enumerate(frame_indices):
            n = dups[ix]
            res.extend([e] * n)
        return res


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        if len(frame_indices) < self.size:
            # duplicated(allow duplicated elemnts) sampling (random)
            frame_indices = RandomDuplicatedSampling(self.size)(frame_indices)

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]
        return out


def to_normalized_float_tensor(vid):
    vid = torch.stack([torch.as_tensor(np.array(pic)) for pic in vid])
    return vid.to(torch.float32) / 255


def normalize(vid, mean, std):
    shape = (1,) * (vid.dim() - 1) + (-1,)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


def denormalize(vid, mean, std):
    shape = (1,) * (vid.dim() - 1) + (-1,)
    mean = torch.as_tensor(mean).reshape(shape).to(vid.device)
    std = torch.as_tensor(std).reshape(shape).to(vid.device)
    return vid * std + mean


class MultiScaleCornerCrop3D(object):
    def __init__(self, transform2D):
        self.transform2D = transform2D

    def __call__(self, vid):
        out = []
        for pic in vid:
            if isinstance(pic, torch.Tensor):
                if pic.dtype == torch.uint8:
                    scale = 1
                elif pic.dtype == torch.float32:
                    scale = 255
                pic = tf_func.to_pil_image((pic.float().cpu() * scale).byte())
            elif isinstance(pic, np.ndarray):
                pic = Image.fromarray(pic)

            crop = self.transform2D(pic)
            out.append(crop)

        return out

    def randomize_parameters(self):
        self.transform2D.randomize_parameters()


import numbers


class Resize3D(object):
    def __init__(self, size, keep_ratio=True, interpolation_mode="bilinear"):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
        self.size = size
        self.keep_ratio = keep_ratio
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        size, scale = None, None
        if isinstance(self.size, numbers.Number):
            if self.keep_ratio:
                scale = self.size / min(clip.shape[-2:])
            else:
                size = (int(self.size), int(self.size))
        else:
            if self.keep_ratio:
                scale = min(
                    self.size[0] / clip.shape[-2], self.size[1] / clip.shape[-1]
                )
            else:
                size = self.size
        clip = nn.functional.interpolate(
            clip,
            size=size,
            scale_factor=scale,
            mode=self.interpolation_mode,
            align_corners=False,
        )

        return list(clip)

    def randomize_parameters(self):
        pass


class RandomCrop3D(object):
    def __init__(self, transform2D):
        self.transform2D = transform2D

    def __call__(self, vid):
        out = []
        for pic in vid:
            if isinstance(pic, torch.Tensor):
                if pic.dtype == torch.uint8:
                    scale = 1
                elif pic.dtype == torch.float32:
                    scale = 255
                pic = tf_func.to_pil_image((pic.float().cpu() * scale).byte())
            elif isinstance(pic, np.ndarray):
                pic = Image.fromarray(pic)

            crop = tf_func.crop(pic, *self.crop_position)
            out.append(crop)

        return out

    def randomize_parameters(self, frame):
        self.crop_position = self.transform2D.get_params(frame, self.transform2D.size)


class RandomResizedCrop3D(object):
    def __init__(self, transform2D):
        self.transform2D = transform2D

    def __call__(self, vid):
        # vid : (T,H,W,C)
        out = []
        for pic in vid:
            if isinstance(pic, torch.Tensor):
                if pic.dtype == torch.uint8:
                    scale = 1
                elif pic.dtype == torch.float32:
                    scale = 255
                pic = tf_func.to_pil_image((pic.float().cpu() * scale).byte())
            elif isinstance(pic, np.ndarray):
                pic = Image.fromarray(pic)

            crop = tf_func.resized_crop(
                pic, *self.crop_position, size=self.transform2D.size
            )
            out.append(crop)

        return out

    def randomize_parameters(self, frame):
        self.crop_position = self.transform2D.get_params(
            frame, self.transform2D.scale, self.transform2D.ratio
        )


class RandomRotation3D(object):
    def __init__(self, transform2D):
        self.transform2D = transform2D

    def __call__(self, vid):
        # vid : Tensor (T,H,W,C)
        out = []
        for pic in vid:
            if isinstance(pic, torch.Tensor):
                if pic.dtype == torch.uint8:
                    scale = 1
                elif pic.dtype == torch.float32:
                    scale = 255
                pic = tf_func.to_pil_image((pic.float().cpu() * scale).byte())
            elif isinstance(pic, np.ndarray):
                pic = Image.fromarray(pic)
            rotated = tf_func.rotate(pic, self.angle)
            out.append(rotated)

        return out

    def randomize_parameters(self):
        self.angle = self.transform2D.get_params(self.transform2D.degrees)


class RandomHorizontalFlip3D(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        # vid : Tensor (T,H,W,C)
        out = []
        for pic in vid:
            if isinstance(pic, torch.Tensor):
                if pic.dtype == torch.uint8:
                    scale = 1
                elif pic.dtype == torch.float32:
                    scale = 255
                pic = tf_func.to_pil_image((pic.float().cpu() * scale).byte())
            elif isinstance(pic, np.ndarray):
                pic = Image.fromarray(pic)

            if self.num < self.p:
                pic = tf_func.hflip(pic)

            out.append(pic)

        return out

    def randomize_parameters(self):
        self.num = random.random()


class CenterCrop3D(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        out = []
        for pic in vid:
            if isinstance(pic, torch.Tensor):
                if pic.dtype == torch.uint8:
                    scale = 1
                elif pic.dtype == torch.float32:
                    scale = 255
                pic = tf_func.to_pil_image((pic.float().cpu() * scale).byte())
            elif isinstance(pic, np.ndarray):
                pic = Image.fromarray(pic)

            crop = tf_func.center_crop(pic, self.size)
            out.append(crop)

        return out

    def randomize_parameters(self):
        pass


class ToTensor3D(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

    def randomize_parameters(self):
        pass


class Normalize3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)

    def randomize_parameters(self):
        pass
