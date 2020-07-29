import torch
import torchvision
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import Dataset
import pandas as pd
import os
from utils.transforms import Resize3D, LoopPadding
from ._utils import *
from PIL import Image

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import unfold

import cv2
import numpy as np
import re


class UCF101(Dataset):
    def __init__(
        self,
        root,
        annotation_path,
        detection_file_path,
        sample_unit="video",
        num_workers=1,
        sample_rate=5,
        input_size=(16, 128, 171),
        train=True,
        hard_augmentation=True,
        fold=1,
        temporal_transform=None,
        spatial_transform=None,
        norm_method=None,
    ):

        super().__init__()

        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))
        if sample_unit not in ["clip", "video"]:
            raise ValueError(
                "invalid sample_unit argument; sample unit should be in (`clip` | `video`)"
            )
        self.root = root  # RGB frame root
        self.flow_root = os.path.join(os.path.dirname(root), "tvl1_flow")

        """
            detection data with Mask-RCNN
            powered by https://github.com/ayoolaolafenwa/PixelLib
        """
        self.detection_data = pd.read_csv(
            detection_file_path,
            sep=" ",
            names=["object_class", "x_center", "y_center", "width", "height"],
        )

        extensions = ("avi",)
        self.fold = fold
        self.train = train
        self.duration, *self.img_size = input_size
        self.sample_rate = sample_rate
        self.sample_unit = sample_unit

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None
        )
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.indices = self._select_fold(video_list, annotation_path, fold, train)

        if self.sample_unit == "clip":
            # validation is performed for each clip
            self.video_clips = VideoClips(
                video_list,
                clip_length_in_frames=self.duration,
                frames_between_clips=self.duration,
                num_workers=num_workers,
                frame_rate=None if sample_rate == 1 else sample_rate,
            ).subset(self.indices)

        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.norm_method = norm_method
        self.hard_augmentation = hard_augmentation

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
        indices = [
            i
            for i in range(len(video_list))
            if video_list[i][len(self.root) + 1 :] in selected_files
        ]
        return indices

    def fetch_video(self, video_idx):
        video_path, label = self.samples[self.indices[video_idx]]
        video, audio, info = torchvision.io.read_video(video_path)
        clip_pts = (
            torch.as_tensor(list(range(len(video)))[:: self.sample_rate])
            / self.sample_rate
        )

        # temporal random crop
        if self.temporal_transform is not None:
            if self.train:
                clip_pts = self.temporal_transform(clip_pts.tolist())

        # convert List -> Tensor
        clip_pts = torch.as_tensor(clip_pts)
        video = video[:: self.sample_rate][clip_pts]

        return video, label, clip_pts, video_path

    def fetch_clip(self, clip_idx):
        clip, audio, info, video_idx, clip_pts = self.video_clips.get_clip(
            clip_idx
        )  # clip_pts-'frame index'
        video_path, label = self.samples[self.indices[video_idx]]
        clip_pts /= self.sample_rate
        if self.sample_rate == 1:
            clip_pts -= 1
            clip_pts = torch.clamp(clip_pts, 0, len(clip_pts) - 1)

        return clip, label, clip_pts, video_path

    def fetch_data(self, idx):
        # train : fetch entire video -> randomly select frames to get a clip (for augmentation) -> sample a clip per each video
        # val/test : fetch a clip -> w/o random frame selection -> sample multiple clips per each video
        video, label, clip_pts, video_path = (
            self.fetch_clip(idx)
            if self.sample_unit == "clip"
            else self.fetch_video(idx)
        )

        video = video.permute(0, 3, 1, 2)

        video = Resize3D(
            size=self.img_size, interpolation=Image.BILINEAR, to_tensor=True
        )(video)
        query = os.path.splitext(video_path[len(self.root.rstrip("/")) + 1 :])[0]
        query = [
            os.path.join(query, "thumb{:04d}.txt".format(i + 1)) for i in clip_pts
        ]  # starts from *0001.txt

        detection_res = self.detection_data.loc[query].fillna(
            0.0
        )  # zero-fill for non-detected frame
        # use only person class detection result
        detection_res_filtered = detection_res.apply(
            lambda x: pd.Series([0.0] * len(x), index=x.index) if x[0] > 0 else x,
            axis=1,
        )

        del detection_res_filtered["object_class"]  # drop object_class column
        mask = []
        flow = []
        C, H, W = video[0].shape
        for q in query:
            # mask
            m = generate_maskImg(detection_res_filtered, q, W, H)
            mask.append(m)

            _, vid, fid = q.split("/")
            vid = vid.replace("HandStandPushups", "HandstandPushups")
            
            fnbrRegex = re.compile(r"\d{4}")
            fnbr = max(1, eval(fnbrRegex.findall(fid)[0].lstrip("0"))-1)
            fid = f"frame{fnbr:06}.jpg"

            # flow
            flo_stack = []
            for comp in list("uvz"):
                if comp != 'z':
                    flo_path = os.path.join(self.flow_root, comp, vid, fid)
                    flo = Image.open(flo_path).convert("L").resize(
                        self.img_size[::-1], resample=Image.BILINEAR
                    )
                    e = torchvision.transforms.ToTensor()(flo)
                else:
                    e = torch.zeros_like(e)
                flo_stack.append(e)
            
            flo_stack = torch.cat(flo_stack, dim=0)
            flow.append(flo_stack)

        # conver label to tensor
        label = torch.tensor(label).long()

        return video, mask, flow, label, clip_pts

    def stack_multiple_clips(self, video, mask, flow, clip_pts):
        # convert to tensor for smart indexing
        video = torch.stack(video)
        mask = torch.stack(mask)
        flow = torch.stack(flow)

        clip_pts = torch.as_tensor(LoopPadding(size=self.duration)(clip_pts.tolist()))

        multi_clip_pts = self.temporal_transform(clip_pts)
        video_stack = []
        mask_stack = []
        flow_stack = []
        for sample_pts in multi_clip_pts:
            sample_pts = sample_pts.tolist()
            # convert List -> Tensor
            sample_pts = torch.as_tensor(sample_pts)
            video_stack.append(video[sample_pts])  # (D,C,H,W)
            mask_stack.append(mask[sample_pts])  # (D,1,H,W)
            flow_stack.append(
                decompose_flow_frames(
                    flow[sample_pts].permute(1, 0, 2, 3), transpose=False
                )
            )  # (N,2,H,W)

        # stack clips
        try:
            video_stack = torch.stack(video_stack)  # (nclips,D,C,H,W)
            mask_stack = torch.stack(mask_stack)  # (nclips,D,1,H,W)
            flow_stack = torch.stack(flow_stack)  # (nclips,D,2,H,W)
        except:
            import ipdb

            ipdb.set_trace()

        return video_stack, mask_stack, flow_stack

    def __len__(self):
        if self.sample_unit == "clip":
            return self.video_clips.num_clips()
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        video, mask_frames, flow_frames, label, clip_pts = self.fetch_data(idx)

        permute_axis = (3, 0, 1, 2)
        if not self.train:
            if self.sample_unit == "video":
                # testing, but sample_unit = video => multiple logit for a each video, which will be averaged later
                video, mask_frames, flow_frames = self.stack_multiple_clips(
                    video, mask_frames, flow_frames, clip_pts
                )
                permute_axis = (0, 4, 1, 2, 3)
        if self.spatial_transform is not None:
            if self.hard_augmentation:
                aug_result = self.spatial_transform(
                    frames=video, mask_frames=mask_frames, flow_frames=flow_frames
                )
                video, mask_frames, flow_frames = tuple(
                    [
                        aug_result.get(k)
                        for k in ["image_frames", "mask_frames", "flow_frames"]
                    ]
                )
            else:
                # init random vars
                self.spatial_transform.randomize_parameters(video)
                video = self.spatial_transform(video)
                mask_frames = self.spatial_transform(mask_frames)
                flow_frames = self.spatial_transform(flow_frames)

            video = self.norm_method(video).permute(*permute_axis)  # apply norm method
            mask_frames = mask_frames[..., :1].permute(*permute_axis)
            flow_frames = flow_frames.permute(*permute_axis)

            if len(permute_axis) == 4:
                flow_frames = decompose_flow_frames(flow_frames)

        return video, mask_frames, flow_frames, label
