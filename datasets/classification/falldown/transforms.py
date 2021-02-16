import copy
from inspect import signature
import torch
from .utils_cv.action_recognition.references import functional_video as F
from .utils_cv.action_recognition.references.transforms_video import *
import random

# "ResizeVideo"                 [ok],
# "RandomCropVideo"             [ok],
# "RandomResizedCropVideo"      [ok],
# "CenterCropVideo"             [ok],
# "NormalizeVideo"              [ok],
# "ToTensorVideo"               [ok],
# "RandomHorizontalFlipVideo"   [ok],


def get_sync_transforms(transforms):
    tfms = []
    for t in transforms.transforms:
        sig = signature(t.__init__)
        params = sig.parameters.keys()
        params = {p: getattr(t, p) for p in params}
        sync_t = eval("Sync" + t.__class__.__name__)(**params)
        tfms.append(sync_t)

    return SyncCompose(tfms)


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


class SyncCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip, mask):
        for t in self.transforms:
            clip, mask = t(clip, mask)
        return clip, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class SyncResizeVideo(ResizeVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_resize = ResizeVideo(*args, **kwargs)

        # use nearest interpolation for mask
        # kwargs["interpolation_mode"] = "nearest"
        self.mask_resize = ResizeVideo(*args, **kwargs)

    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        # return result with the same transformation
        return [self.clip_resize(clip), self.mask_resize(mask)]


class SyncNormalizeVideo(NormalizeVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = NormalizeVideo(*args, **kwargs)

    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        # return result with the same transformation
        return [
            self.transform(clip),
            self.transform(mask),
        ]


class SyncCenterCropVideo(CenterCropVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = CenterCropVideo(*args, **kwargs)

    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        # return result with the same transformation
        return [self.transform(clip), self.transform(mask)]


# create custom class transform
class SyncToTensorVideo(ToTensorVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = ToTensorVideo(*args, **kwargs)

    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        # return result with the same transformation
        return [self.transform(clip), self.transform(mask)]


class SyncRandomCropVideo(RandomCropVideo):
    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        # fix parameter
        i, j, h, w = self.get_params(clip, self.size)

        # return result with the same transformation
        return [F.crop(clip, i, j, h, w), F.crop(mask, i, j, h, w)]


class SyncRandomResizedCropVideo(RandomResizedCropVideo):
    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        # fix parameter
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)

        # return result with the same transformation
        return [
            F.resized_crop(clip, i, j, h, w, self.size,
                           self.interpolation_mode),
            F.resized_crop(mask, i, j, h, w, self.size,
                           self.interpolation_mode),
            #    interpolation_mode="nearest"),
        ]


class SyncRandomHorizontalFlipVideo(RandomHorizontalFlipVideo):
    def __call__(self, clip, mask):
        assert clip.size() == mask.size()
        is_flip = False
        # fix parameter
        if random.random() < self.p:
            is_flip = True

        # apply same transformation
        if is_flip:
            clip = F.hflip(clip)
            mask = F.hflip(mask)

        return [clip, mask]
