from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
import os
import glob
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import tqdm
import torchvision.transforms as TF
from utils.transforms import Resize3D


class BenchmarkDS(torch.utils.data.Dataset):
    def __init__(self, localizer, frames_per_clip):
        self.localizer = localizer
        self.frames_per_clip = frames_per_clip

    def getMask(self, video):
        T, H, W = video.shape[:3]

        mask_img = torch.zeros((1, T, H, W), dtype=torch.float32)
        for res in self.localizer.detect_from_video(video=video):
            pos_list, idx = res.person_pos, res.idx
            for pos in pos_list:
                bx, by, bw, bh = pos
                mask_img[:, idx % self.frames_per_clip,
                         int(by-bh/2):int(by+bh/2),
                         int(bx-bw/2):int(bx+bw/2)] = 1.0

        return mask_img


class UCF101(torch.utils.data.Dataset):
    def __init__(self,
                 spatial_transform,
                 gait_imgSize, yolo_imgSize,
                 *args, **kwargs):
        self.data = torchvision.datasets.UCF101(*args, **kwargs)

        self.gait_imgSize = gait_imgSize
        self.yolo_imgSize = yolo_imgSize

        self.spatial_transform = None
        if spatial_transform is not None:
            self.spatial_transform = spatial_transform

    def transform(self, inputs, randomize=True):
        if randomize:
            first_frame = Image.fromarray(inputs[0].cpu().numpy())
            self.spatial_transform.randomize_parameters(first_frame)
        inputs = self.spatial_transform(inputs)

        return inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs0, _, _, video_ix = self.data.video_clips.get_clip(index)
        vids, targets = self.data.samples[self.data.indices[video_ix]]

        def resize_video(vid, size):
            res = []
            for t, f in enumerate(vid):
                f = cv2.resize(f.numpy(), size)
                res.append(torch.as_tensor(f))
            return torch.stack(res)

        # Resize video before transformation! to prevent extensive noise generation.
        inputs0 = resize_video(inputs0, self.gait_imgSize)

        if self.spatial_transform is not None:
            inputs0 = self.transform(inputs0, randomize=True)

        h0, w0 = inputs0.shape[1:3]

        # inputs for yolov3..
        inputs1 = resize_video(inputs0, self.yolo_imgSize)
        h1, w1 = inputs1.shape[1:3]

        shapes = torch.tensor(((((h0, w0)), ((h1/h0, w1/w0)))))

        inputs0 = inputs0.permute(3, 0, 1, 2)
        inputs1 = inputs1.permute(3, 0, 1, 2)

        return inputs0, inputs1, targets, vids, shapes


class FallDataset(VisionDataset):
    """
    `URFD <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    URFD is an fall detection video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the URFD Dataset.
        annotation_path (str): path to the folder containing the annotation files
        detection_file_path (str): path to a detection result pkl file
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 to 5.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        preCrop (bool, optional): if ``True``, apply pre-cropping to input image to get local image
    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        mask (Tensor[T, H, W, 1]): the `T' mask frames, representing person bbox in intereset
        label (int): class of the video clip (fall / adl); ADL - Activities of Daily Living
    """

    def __init__(self, root, annotation_path, detection_file_path, frames_per_clip, step_between_clips=1,
                 frame_rate=None, fold=1, train=True, spatial_transform=None, temporal_transform=None, norm_method=None, preCrop=None, img_size=112,
                 _precomputed_metadata=None, num_workers=1, _video_width=0,
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(FallDataset, self).__init__(root)
        if not 1 <= fold <= 5:
            raise ValueError(
                "fold should be between 1 and 3, got {}".format(fold))

        extensions = ('avi',)
        self.fold = fold
        self.train = train
        self.frames_per_clip = frames_per_clip
        self.preCrop = preCrop

        self.detection_data = pd.read_pickle(detection_file_path)

        # eval string values
        self.detection_data.idx = self.detection_data.idx.map(
            lambda x: eval(x))
        self.detection_data.pos = self.detection_data.pos.map(
            lambda x: [int(e) for e in eval(x)])

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.indices = self._select_fold(
            video_list, annotation_path, fold, train)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.norm_method = norm_method

        self.img_size = img_size

    def apply_spatial_transform(self, inputs, randomize=True, normalize=True):
        if randomize:
            self.spatial_transform.randomize_parameters(inputs)
        inputs = self.spatial_transform(inputs)
        if normalize:
            inputs = self.norm_method(inputs)

        return inputs

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i][len(
            self.root) + 1:] in selected_files]
        return indices

    def fetch_data(self, idx):
        video_path, label = self.samples[self.indices[idx]]
        video, audio, info = torchvision.io.read_video(video_path)
        clip_pts = list(range(len(video)))

        return video, label, clip_pts, video_path

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        video, label, clip_pts, video_path = self.fetch_data(idx)
        video_name = os.path.basename(video_path).rstrip('.avi')

        if self.temporal_transform is not None:
            clip_pts = self.temporal_transform(clip_pts)
        # convert to tensor
        clip_pts = torch.as_tensor(clip_pts)

        # corresponding masks
        cur_detection = self.detection_data[self.detection_data.vids == video_name]
        overlap = cur_detection[cur_detection.idx.isin(clip_pts)]

        masks = torch.zeros_like(video)
        _, detecionTimeStamps, bboxPositions = overlap.values.T
        for t, (xmin, ymin, xmax, ymax) in zip(detecionTimeStamps, bboxPositions):
            masks[t, ymin:ymax, xmin:xmax] = 255

        # smart indexing, retrieve a subVideoClip
        video = video[clip_pts]
        masks = masks[clip_pts]

        # reize frames
        video = Resize3D(size=self.img_size,
                         interpolation=Image.BILINEAR)(video)
        masks = Resize3D(size=self.img_size,
                         interpolation=Image.NEAREST)(masks)

        if self.spatial_transform is not None:
            video = self.apply_spatial_transform(
                video, randomize=True, normalize=True)
            masks = self.apply_spatial_transform(
                masks, randomize=False, normalize=False)

        permute_axis = (3, 0, 1, 2) if self.train else (0, 4, 1, 2, 3)

        # (T,H,W,C) => (C,T,H,W)
        video = video.permute(*permute_axis)
        masks = masks.permute(*permute_axis)

        masks = masks[:1] if self.train else masks[:, :1]
        valid_len = video.size(1) if self.train else video.size(2)

        return video, masks, label, video_name, valid_len
