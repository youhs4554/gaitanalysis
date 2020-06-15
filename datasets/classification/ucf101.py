import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
from utils.transforms import Resize3D, LoopPadding
from ._utils import *
from PIL import Image

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import unfold


class UCF101(Dataset):
    def __init__(self,
                 root, annotation_path, detection_file_path,
                 sample_rate=5, input_size=(16, 128, 171), train=True, fold=1,
                 temporal_transform=None, spatial_transform=None, norm_method=None):

        super().__init__()

        if not 1 <= fold <= 3:
            raise ValueError(
                "fold should be between 1 and 3, got {}".format(fold))
        self.root = root
        # detection data with Mask-RCNN
        # powered by https://github.com/ayoolaolafenwa/PixelLib
        self.detection_data = pd.read_csv(detection_file_path,
                                          sep=' ',
                                          names=["object_class", "x_center", "y_center", "width", "height"])

        extensions = ('avi',)
        self.fold = fold
        self.train = train

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.indices = self._select_fold(
            video_list, annotation_path, fold, train)

        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.norm_method = norm_method
        self.duration, *self.img_size = input_size
        self.sample_rate = sample_rate

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

    def fetch_video(self, idx):
        video_path, label = self.samples[self.indices[idx]]
        video, audio, info = torchvision.io.read_video(video_path)
        clip_pts = torch.as_tensor(list(range(len(video)))[
                                   ::self.sample_rate]) / self.sample_rate

        # temporal crop
        # @train        : random
        # @val/test     : None
        if self.temporal_transform is not None:
            clip_pts = self.temporal_transform(clip_pts.tolist())

        # convert List -> Tensor
        clip_pts = torch.as_tensor(clip_pts)
        video = video[::self.sample_rate][clip_pts].permute(0, 3, 1, 2)

        video = Resize3D(size=self.img_size,
                         interpolation=Image.BILINEAR, to_tensor=True)(video)
        query = os.path.splitext(video_path[len(self.root.rstrip('/'))+1:])[0]
        query = [os.path.join(
            query, "thumb{:04d}.txt".format(i+1)) for i in clip_pts]  # starts from *0001.txt

        detection_res = self.detection_data.loc[query].fillna(
            0.0)  # zero-fill for non-detected frame
        # use only person class detection result
        detection_res_filtered = detection_res.apply(lambda x: pd.Series(
            [0.0]*len(x), index=x.index) if x[0] > 0 else x, axis=1)

        del detection_res_filtered["object_class"]  # drop object_class column
        mask = []
        C, H, W = video[0].shape
        for q in query:
            m = generate_maskImg(detection_res_filtered,
                                 q, W, H)
            mask.append(m)

        # conver label to tensor
        label = torch.tensor(label).long()

        return video, mask, label, clip_pts

    def stack_multiple_clips(self, video, mask, clip_pts):
        # convert to tensor for smart indexing
        video = torch.stack(video)
        mask = torch.stack(mask)

        clip_pts = torch.as_tensor(LoopPadding(
            size=self.duration)(clip_pts.tolist()))

        multi_clip_pts = unfold(clip_pts, self.duration, self.duration)
        video_stack = []
        mask_stack = []
        for sample_pts in multi_clip_pts:
            sample_pts = LoopPadding(size=self.duration)(sample_pts.tolist())
            # convert List -> Tensor
            sample_pts = torch.as_tensor(sample_pts)
            video_stack.append(video[sample_pts])  # (D,C,H,W)
            mask_stack.append(mask[sample_pts])  # (D,1,H,W)

        # stack clips
        video_stack = torch.stack(video_stack)  # (nclips,D,C,H,W)
        mask_stack = torch.stack(mask_stack)    # (nclips,D,1,H,W)

        return video_stack, mask_stack

    def apply_spatial_transform(self, video, randomize=True, normalize=True):
        if randomize:
            self.spatial_transform.randomize_parameters(video)
        video = self.spatial_transform(video)
        if normalize:
            video = self.norm_method(video)

        return video

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        video, mask, label, clip_pts = self.fetch_video(idx)

        permute_axis = (3, 0, 1, 2)
        if not self.train:
            video, mask = self.stack_multiple_clips(video, mask, clip_pts)
            permute_axis = (0, 4, 1, 2, 3)

        if self.spatial_transform is not None:
            video = self.apply_spatial_transform(
                video, randomize=True, normalize=True).permute(*permute_axis)
            mask = self.apply_spatial_transform(
                mask, randomize=False, normalize=False)[..., :1].permute(*permute_axis)

        return video, mask, label
