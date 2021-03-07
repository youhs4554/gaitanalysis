# %%
import math
from numpy.random import randint
import warnings
import matplotlib.pyplot as plt
import tqdm
from .transforms import *
from PIL import Image, ImageDraw
import decord
import torch
from typing import List, Tuple, Union
from inspect import signature
from .utils_cv.action_recognition.dataset import (
    VideoDataset,
    DEFAULT_MEAN,
    DEFAULT_STD, VideoRecord,
)
from sklearn.model_selection import StratifiedKFold
import natsort
import os
import glob
from os.path import join
import numpy as np


def split_train_test_for_multicam(root):

    split_file_root = root.replace("video", "TrainTestlist")
    os.system("mkdir -p {}".format(split_file_root))

    n_cameras = 8
    class2ix = {"adl": 0, "fall": 1}

    for cam_id in range(1, n_cameras+1):
        test_split = []
        train_split = []
        test_split_file_path = os.path.join(
            split_file_root, "testlist{:02d}.txt".format(cam_id))
        train_split_file_path = os.path.join(
            split_file_root, "trainlist{:02d}.txt".format(cam_id))

        video_files = glob.glob(root + "/*/*")
        for vf in video_files:
            basename = os.path.basename(vf)
            vid = os.path.splitext(vf[len(root)+1:])[0]
            class_ix = str(class2ix.get(os.path.dirname(vid)))
            if basename.split("-")[1] == "cam" + str(cam_id):
                test_split.append(" ".join([vid, class_ix]))
            else:
                train_split.append(" ".join([vid, class_ix]))

        np.savetxt(test_split_file_path, test_split, fmt="%s")
        np.savetxt(train_split_file_path, train_split, fmt="%s")


def split_train_test_for_urfd(root, random_state=0, n_splits=5):
    split_file_root = root.replace("video", "TrainTestlist")
    os.system("mkdir -p {}".format(split_file_root))

    video_dirs = natsort.natsorted(glob.glob(root + '/*/*'))

    np.random.seed(random_state)
    np.random.shuffle(video_dirs)

    class2idx = {'adl': 0, 'fall': 1}

    formated_video_dirs = np.array(
        [os.path.splitext(x[len(root.rstrip('/'))+1:])[0] for x in video_dirs])

    kf = StratifiedKFold(n_splits)

    for k, (train_ix, test_ix) in enumerate(kf.split(formated_video_dirs, [os.path.dirname(x) for x in formated_video_dirs])):
        print()

        _train, _test = np.array(formated_video_dirs)[
            train_ix], np.array(formated_video_dirs)[test_ix]

        _train_lab = [os.path.dirname(x) for x in _train]
        _test_lab = [os.path.dirname(x) for x in _test]
        print()
        print(
            f'[splist-{k}] train : {np.unique(_train_lab, return_counts=True)}, test : {np.unique(_test_lab, return_counts=True)}')
        for _split, _data in zip(['train', 'test'], [_train, _test]):
            lines = []
            for i in range(len(_data)):
                line = _data[i] + " "
                line += str(class2idx[os.path.dirname(_data[i])])

                lines.append(line + '\n')

            with open(os.path.join(split_file_root, f"{_split}list{k+1:02d}.txt"), 'w') as fp:
                fp.writelines(lines)


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


def generate_maskImg(detection_res, W, H):
    _, x_center, y_center, box_width, box_height = detection_res.T
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
    mask = Image.new("RGB", size=(W, H))
    for x in res:
        ImageDraw.Draw(mask).rectangle(x, fill="white")

    # to array
    mask = np.array(mask)
    return mask


class ActivityRecogPlusSegmentationDataset(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def frame_home(self) -> str:
        frame_home = os.path.join(os.path.dirname(self.root), "frames")

        assert os.path.exists(
            frame_home), "ValueError : frame_home does not exist"
        return frame_home

    def generate_mask_frames(self, vid, offset, num_frames, w, h):
        mask_frames = []
        for ix in range(offset, offset + self.sample_length):
            ix = min(ix + 1, num_frames - 1)  # max clipping
            detection_file = os.path.join(vid, f"thumb{ix:05d}.txt")
            if open(detection_file).read() == "":
                # empty detection file
                detection_res = np.zeros((5,))
            else:
                detection_res = np.loadtxt(detection_file)
            if detection_res.ndim == 1:
                detection_res = detection_res[np.newaxis, :]
            person_detection_res = detection_res[np.where(
                detection_res[:, 0] == 0)]
            # if len(person_detection_res):
            #     # select most significant person
            #     person_detection_res = person_detection_res[[0], :]
            # mask_frame from detection result
            m = generate_maskImg(person_detection_res, w, h)
            mask_frames.append(m)

        mask_frames = np.array(mask_frames)
        return mask_frames

    def _sample_indices(self, record: VideoRecord) -> List[int]:
        """
        Create a list of frame-wise offsets into a video record. Depending on
        whether or not 'random shift' is used, perform a uniform sample or a
        random sample.

        Args:
            record (VideoRecord): A video record.

        Return:
            list: Segment offsets (start indices)
        """
        if record.num_frames > self.presample_length:
            if self.random_shift:
                # Random sample
                offsets = np.sort(
                    randint(
                        record.num_frames - self.presample_length + 1,
                        size=self.num_samples,
                    )
                )

                # # TSN
                # average_duration = (
                #     record.num_frames - self.presample_length + 1) // self.num_samples
                # if average_duration > 0:
                #     offsets = np.multiply(list(range(
                #         self.num_samples)), average_duration) + randint(average_duration, size=self.num_samples)
                # elif record.num_frames > self.num_samples:
                #     offsets = np.sort(
                #         randint(record.num_frames - self.presample_length + 1, size=self.num_samples))
                # else:
                #     offsets = np.zeros((self.num_samples,))

            else:
                # # Uniform sample
                # distance = (
                #     record.num_frames - self.presample_length + 1
                # ) / self.num_samples

                # offsets = np.array(
                #     [
                #         int(distance / 2.0 + distance * x)
                #         for x in range(self.num_samples)
                #     ]
                # )

                # # sliding window (winsize == sample_length)
                # n_windows = int((record.num_frames -
                #                  self.presample_length + 1) / self.sample_length + 1)
                # offsets = np.array(
                #     [
                #         i * self.sample_length for i in range(n_windows)
                #     ]
                # )

                # sliding window (winsize == sample_step)
                n_windows = math.floor((record.num_frames-self.sample_length) /
                                       self.sample_step+1)
                offsets = np.array(
                    [
                        i * self.sample_step for i in range(n_windows)
                    ]
                )

        else:
            if self.warning:
                warnings.warn(
                    f"num_samples and/or sample_length > num_frames in {record.path}"
                )
            offsets = np.zeros((self.num_samples,), dtype=int)

        return offsets

    def sample_clip_from_video(
        self, idx: int
    ) -> Union[
        Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor, int]
    ]:
        record = self.video_records[idx]
        vid = os.path.join(self.frame_home, record.path)

        video_reader = decord.VideoReader(
            "{}.{}".format(os.path.join(
                self.root, record.path), self.video_ext),
            # TODO try to add `ctx=decord.ndarray.gpu(0) or .cuda(0)`
        )
        record._num_frames = len(video_reader)

        offsets = self._sample_indices(record)

        clips = np.array([self._get_frames(video_reader, o) for o in offsets])
        h, w = clips.shape[2:4]
        mask_frames = np.array(
            [
                self.generate_mask_frames(vid, o, record._num_frames, w, h)
                for o in offsets
            ]
        )

        sync_transforms = get_sync_transforms(self.transforms)

        res = []
        if self.random_shift and clips.shape[0] == 1:
            # apply same transforms on different inputs(rgb & mask)
            rgb, mask = sync_transforms(
                torch.from_numpy(clips[0]), torch.from_numpy(mask_frames[0])
            )
            mask = F.denormalize(mask, DEFAULT_MEAN, DEFAULT_STD)[[0], :, :, :]
            res += [rgb, mask, torch.tensor(record.label)]
        else:
            # [S, T, H, W, C] -> [S, C, T, H, W]
            rgb_stack, mask_stack = torch.stack(
                [
                    torch.stack(
                        sync_transforms(torch.from_numpy(c),
                                        torch.from_numpy(m))
                    )
                    for c, m in zip(clips, mask_frames)
                ]
            ).transpose(0, 1)

            mask_stack = torch.stack(
                [F.denormalize(m, DEFAULT_MEAN, DEFAULT_STD)
                 for m in mask_stack]
            )[:, [0], :, :, :]

            res += [rgb_stack, mask_stack, torch.tensor(record.label)]

        return tuple(res)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor, int]
    ]:
        if getattr(self, "samples", None) is not None:
            return self.samples[idx]
        else:
            return self.sample_clip_from_video(idx)

    def show_batch(self, train_or_test: str = "train", rows: int = 2) -> None:
        """Plot first few samples in the datasets"""
        if train_or_test == "train":
            batch = [self.train_ds[i] for i in range(rows)]
        elif train_or_test == "test":
            batch = [self.test_ds[i] for i in range(rows)]
        else:
            raise ValueError("Unknown data type {}".format(train_or_test))

        images = [item[0] for item in batch]
        masks = [item[1].repeat(3, 1, 1, 1) for item in batch]
        labels = [item[2] for item in batch]

        self._show_batch(images, labels, self.sample_length)
        self._show_batch(masks, labels, self.sample_length)


def URFD(dataset_root, fold, sample_length, batch_size, tsn=False):
    split_file_root = dataset_root.replace("video", "TrainTestlist")
    if not os.path.exists(split_file_root):
        # 5-fold stratified cv
        split_train_test_for_urfd(dataset_root, n_splits=5)

    train_split_file = os.path.join(
        split_file_root, "trainlist{:02d}.txt".format(fold))
    test_split_file = os.path.join(
        split_file_root, "testlist{:02d}.txt".format(fold))

    return ActivityRecogPlusSegmentationDataset(root=dataset_root,
                                                train_split_file=train_split_file,
                                                test_split_file=test_split_file,
                                                video_ext="avi",
                                                sample_length=sample_length,
                                                test_sample_step=1,
                                                batch_size=batch_size,
                                                num_samples=3 if tsn else 1)


def MulticamFD(dataset_root, fold, sample_length, batch_size, tsn=False):
    split_file_root = dataset_root.replace("video", "TrainTestlist")
    if not os.path.exists(split_file_root):
        # leave one out cv based on camera ix
        split_train_test_for_multicam(dataset_root)
    train_split_file = os.path.join(
        split_file_root, "trainlist{:02d}.txt".format(fold))
    test_split_file = os.path.join(
        split_file_root, "testlist{:02d}.txt".format(fold))

    return ActivityRecogPlusSegmentationDataset(root=dataset_root,
                                                train_split_file=train_split_file,
                                                test_split_file=test_split_file,
                                                video_ext="avi",
                                                sample_length=sample_length,
                                                test_sample_step=5,
                                                batch_size=batch_size,
                                                num_samples=3 if tsn else 1)

# # data = MulticamFD(dataset_root="/data/FallDownData/MulticamFD/video",
# #                   fold=1, sample_length=8, batch_size=4, tsn=True)


# data = URFD(dataset_root="/data/FallDownData/URFD/video",
#             fold=1, sample_length=8, batch_size=4, tsn=True)

# # clip, label = data.train_ds[0]
# # print(clip.shape)
# clips, masks, labels = next(iter(data.train_dl))
# print(clips.shape, masks.shape, labels.shape)
# # %%
# for t in range(8):
#     fig = plt.figure()
#     for i, im in enumerate([clips[0, 0, 0, t], masks[0, 0, 0, t]], start=1):
#         ax = fig.add_subplot(1, 2, i)
#         ax.imshow(im)
# # %%

# %%
