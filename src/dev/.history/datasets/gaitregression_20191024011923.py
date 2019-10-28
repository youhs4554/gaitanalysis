from __future__ import print_function, division

import copy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
import sys
import random
import torchvision.transforms as TF
import torchvision.transforms.functional as tf_func
import torch.nn.functional as F
import collections


def get_direction(patient_positions):
    start_pos, end_pos = [patient_positions[0], patient_positions[-1]]
    s_xmin, s_ymin, s_xmax, s_ymax = eval(start_pos)
    e_xmin, e_ymin, e_xmax, e_ymax = eval(end_pos)
    s_c = (s_xmin + s_xmax) / 2, (s_ymin + s_ymax) / 2
    e_c = (e_xmin + e_xmax) / 2, (e_ymin + e_ymax) / 2
    res = 'approaching' if e_c[1] > s_c[1] else 'leaving'

    return res


def pid2vid(pid):
    num, test_id, trial_id = pid.split('_')
    return '_'.join([num, 'test', test_id, 'trial', trial_id])


def vid2pid(vid):
    split = vid.split('_')
    return '_'.join([split[0], split[2], split[4]])


def arrange_vids(vids, seed=0):
    np.random.seed(seed)
    np.random.shuffle(vids)
    return vids


def filter_input_df_with_vids(df, vids):
    return df[df['vids'].isin(vids)]


def filter_target_df_with_vids(df, vids):
    target_ids = [vid2pid(vid) for vid in vids]
    return df.loc[target_ids]


def split_dataset_with_vids(input_df, target_df, vids,
                            test_size=0.2, random_state=42):
    train_vids, test_vids = train_test_split(
        vids, test_size=test_size, random_state=random_state)

    train_X, train_y = filter_input_df_with_vids(
        input_df, train_vids), filter_target_df_with_vids(target_df,
                                                          train_vids)

    test_X, test_y = filter_input_df_with_vids(
        input_df, test_vids), filter_target_df_with_vids(target_df,
                                                         test_vids)

    return train_X, train_y, train_vids, test_X, test_y, test_vids


def prepare_dataset(input_file, target_file,
                    target_columns, chunk_parts, target_transform=None):

    # if data generation is not the case..
    prefix, ext = os.path.splitext(input_file)

    data_frames = []
    for ix in range(chunk_parts):
        partial = prefix + '-' + str(ix) + ext
        data_frames.append(pd.read_pickle(partial))

    # concat all df
    input_df = pd.concat(data_frames)
    input_df.to_pickle(prefix + '-' + 'merged' + ext)  # save input file

    target_df = pd.read_pickle(target_file)[target_columns]

    possible_vids = sorted(list(set(input_df.vids)))

    # reindex tgt data (to filter-out valid vids)
    target_df = target_df.reindex([vid2pid(vid) for vid in possible_vids])

    not_null_ix = np.where(target_df.notnull().values.all(axis=1))[0]
    possible_vids = [pid2vid(pid) for pid in target_df.index[not_null_ix]]

    target_df = target_df.dropna()

    if target_transform:
        scaled_values = target_transform.fit_transform(target_df)
        target_df.loc[:, :] = scaled_values

    # split dataset (train/test)
    train_X, train_y, train_vids, test_X, test_y, test_vids =\
        split_dataset_with_vids(
            input_df, target_df, possible_vids,
            test_size=0.2, random_state=42)

    return dict(train_X=train_X, train_y=train_y, train_vids=train_vids,
                test_X=test_X, test_y=test_y, test_vids=test_vids,
                input_df=input_df, target_df=target_df)


def generate_dataloader_for_crossvalidation(opt, ds, vids,
                                            ds_class,
                                            phase=None,
                                            spatial_transform=None,
                                            temporal_transform=None,
                                            shuffle=True):

    from torch.utils.data import DataLoader
    X, y = filter_input_df_with_vids(
        ds.X, vids), filter_target_df_with_vids(ds.y, vids)

    ds = ds_class(X, y,
                  opt=opt, phase=phase,
                  spatial_transform=spatial_transform,
                  temporal_transform=temporal_transform)

    for i in range(16):
        ds[i]

    # define dataloader
    loader = DataLoader(ds,
                        batch_size=opt.batch_size,
                        shuffle=shuffle,
                        num_workers=opt.n_threads, drop_last=True)

    return loader


def video_loader(data_root, vid, frame_indices, size, mode='PIL'):
    assert type(size) in [tuple, int], 'size should be tuple or int'

    if type(size) == int:
        size = (size, size)

    if mode == 'numpy':
        res = np.load(os.path.join(data_root, vid) + '.npy')
    elif mode == 'PIL':
        res = []
        subdir = os.path.join(data_root, vid)
        for i in frame_indices:
            f = os.path.join(subdir, f"thumb{int(i):04d}.jpg")
            img = Image.open(f).resize(size)
            res.append(img)

    return res


def process_as_tensor_image(imgs, padding, pad_mode):
    imgs = torch.stack([img
                        for img in imgs])

    imgs = imgs.permute(
        1, 0, 2, 3)     # (C,D,H,W)

    if pad_mode == 'replicate':
        # replicated-padding
        imgs = F.pad(imgs.permute(0, 2, 3, 1), padding,
                     mode=pad_mode).permute(0, 3, 1, 2)
    elif pad_mode == 'zeropad':
        # zero padding
        imgs = F.pad(imgs, padding)

    return imgs


def get_input(video_data, spatial_transform, padding, pad_mode, seed, direction):
    input_ = []

    for idx in range(len(video_data)):
        random.seed(seed)
        img = video_data[idx]
        if direction == 'approaching':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)      # flip left-right
        t_img = spatial_transform(img)   # Tensor image (C,H,W)
        input_.append(t_img)

    input_ = process_as_tensor_image(input_, padding, pad_mode)

    return input_


def get_flows(input_imgs,
              mean=[0.43216, 0.394666, 0.37645],
              std=[0.22803, 0.22145, 0.216989]):

    np_imgs = input_imgs.permute(1, 2, 3, 0).numpy()

    def denormalize_img(img):
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        return img

    prev = denormalize_img(np_imgs[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    flows = []
    for i in range(1, len(np_imgs)):
        nxt = denormalize_img(np_imgs[i])
        nxt = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
        flows.append(
            torch.from_numpy(
                cv2.calcOpticalFlowFarneback(
                    prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            )
        )
        prev = nxt

    return torch.stack(flows).permute(3, 0, 1, 2)


def get_mask(patient_positions, crop_position, angle, padding, pad_mode, opt, direction):
    ratio = {
        'height': opt.img_size/opt.raw_h,
        'width': opt.img_size/opt.raw_w
    }

    mask_ = []

    for idx in range(len(patient_positions)):
        pos = patient_positions[idx]
        xmin, ymin, xmax, ymax = eval(pos)

        xmin, xmax = [round(ratio['width']*v) for v in [xmin, xmax]]
        ymin, ymax = [round(ratio['height']*v) for v in [ymin, ymax]]

        mask = np.stack([np.zeros((opt.img_size, opt.img_size)),
                         np.ones((opt.img_size, opt.img_size))])

        mask_imgs = []
        coord = collections.OrderedDict(
            {'fg': 1,
             'bg': 0})

        ch_order = ['fg', 'bg']

        for i in range(len(ch_order)):
            fill_val = coord.get(ch_order[i])
            mask[i, ymin:ymax, xmin:xmax] = fill_val
            mask_img = Image.fromarray(mask[i])
            if direction == 'approaching':
                mask_img = mask_img.transpose(
                    Image.FLIP_LEFT_RIGHT)      # flip left-right
            mask_imgs.append(mask_img)

        mask_.append(
            torch.cat([
                tf_func.to_tensor(
                    tf_func.resized_crop(
                        tf_func.rotate(
                            mask_imgs[i], angle),
                        *crop_position, size=(opt.sample_size, opt.sample_size)
                    )
                ) for i in range(len(ch_order))
            ])
        )

    mask_ = process_as_tensor_image(mask_, padding, pad_mode)

    return mask_


class GAITDataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 opt,
                 spatial_transform=None, temporal_transform=None):

        self.X = X
        self.y = y
        self.vids = arrange_vids(natsorted(list(set(X.vids))), seed=0)
        self.load_pretrained = opt.load_pretrained

        self.sample_duration = opt.sample_duration

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        self.opt = opt

        self.feats_dir = os.path.join(
            os.path.dirname(opt.data_root), 'FeatsArrays', opt.arch)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        if self.load_pretrained:
            inputs = np.load(os.path.join(
                self.feats_dir, vid)+'.npy')
            inputs = inputs[::self.opt.delta]
        else:
            inputs = []
            stacked_arr = np.load(os.path.join(
                self.opt.data_root, vid) + '.npy')

            for cropped in stacked_arr[::self.opt.delta]:
                img = cv2.resize(cropped, self.opt.sample_size[::-1])
                inputs.append(img)

            # zero padding
            inputs = np.pad(inputs, ((0, self.sample_duration - len(inputs)),
                                     (0, 0), (0, 0), (0, 0)),
                            'constant', constant_values=0)

            if self.spatial_transform:
                self.spatial_transform.randomize_parameters()
                inputs = [self.spatial_transform(
                    Image.fromarray(img)) for img in inputs]

            inputs = torch.stack(inputs, 0).permute(1, 0, 2, 3)

        # target is always same!
        targets = torch.tensor(
            self.y.loc[vid2pid(vid)].values, dtype=torch.float32)

        return inputs, targets, vid


class GAITSegRegDataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 opt,
                 phase,
                 spatial_transform=None, temporal_transform=None):

        self.X = X
        self.y = y
        self.vids = arrange_vids(natsorted(list(set(X.vids))), seed=0)
        self.load_pretrained = opt.load_pretrained

        self.sample_duration = opt.sample_duration

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        self.opt = opt
        self.phase = phase

    def __len__(self):
        return len(self.vids)

    def process_sampled_data(self, cur_X, vid):
        if self.temporal_transform:
            indices_sampled = self.temporal_transform(
                list(range(0, len(cur_X), self.opt.delta)))
        else:
            indices_sampled = list(range(0, len(cur_X), self.opt.delta))

        cur_X = cur_X.iloc[indices_sampled]

        frame_indices, patient_positions = cur_X.idx.values, cur_X.pos.values
        video_data = video_loader(
            self.opt.data_root, vid, frame_indices,
            size=self.opt.img_size, mode='PIL')

        direction = get_direction(patient_positions)

        seed = random.randint(-sys.maxsize, sys.maxsize)

        if self.phase == 'train':
            # @ train; fixed rotation angle for entire video frames
            rotation_method, crop_method = self.spatial_transform.transforms[:2]

            random.seed(seed)

            angle = rotation_method.get_params(
                rotation_method.degrees
            )

            random.seed(seed)

            # for fixed cropping for entire video frames
            crop_position = crop_method.get_params(
                video_data[0], crop_method.scale, crop_method.ratio)

        else:
            # @ test; without tilt and croping
            _start = (self.opt.img_size-self.opt.sample_size)//2
            angle = 0.0
            crop_position = (
                _start, _start, self.opt.sample_size, self.opt.sample_size)

        input_imgs = get_input(video_data,
                               self.spatial_transform,
                               padding=(
                                   0, 0,
                                   0, 0,
                                   0, self.sample_duration - len(frame_indices)),
                               pad_mode='zeropad',
                               seed=seed,
                               direction=direction)

        # # optical flow imgs
        # flows = get_flows(input_imgs)

        # # merge rgb img & optical flow through channel dims
        # input_imgs = torch.cat([input_imgs[:, :-1], flows])

        mask_imgs = get_mask(patient_positions,
                             crop_position, angle,
                             padding=(
                                 0, 0,
                                 0, 0,
                                 0, self.sample_duration - len(frame_indices)),
                             pad_mode='zeropad',
                             opt=self.opt,
                             direction=direction)

        return input_imgs, mask_imgs, len(frame_indices)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        cur_X = self.X[self.X.vids == vid]

        input_imgs, mask_imgs, valid_lengths = self.process_sampled_data(
            cur_X, vid)

        # target is always same!
        targets = torch.tensor(
            self.y.loc[vid2pid(vid)].values, dtype=torch.float32)

        return input_imgs, mask_imgs, targets, vid, valid_lengths
