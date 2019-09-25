from __future__ import print_function, division

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted


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
                    target_columns, chunk_parts):

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
                                            input_transform=None,
                                            target_transform=None,
                                            shuffle=True):

    from torch.utils.data import DataLoader
    X, y = filter_input_df_with_vids(
        ds.X, vids), filter_target_df_with_vids(ds.y, vids)

    ds = ds_class(X, y,
                  opt=opt,
                  input_transform=input_transform,
                  target_transform=target_transform)

    # define dataloader
    loader = DataLoader(ds,
                        batch_size=opt.batch_size,
                        shuffle=shuffle,
                        num_workers=opt.n_threads)

    return loader


def video_loader(data_root, vid, frame_indices, mode='PIL'):
    if mode == 'numpy':
        res = np.load(os.path.join(data_root, vid) + '.npy')
    elif mode == 'PIL':
        res = []
        subdir = os.path.join(data_root, vid)
        for i in frame_indices:
            f = os.path.join(subdir, f"thumb{int(i):04d}.jpg")
            img = Image.open(f).resize((256, 256))
            res.append(img)

    return res


def process_as_tensor_image(imgs, padding):
    imgs = torch.stack([img
                        for img in imgs])

    imgs = imgs.permute(
        1, 0, 2, 3)     # (C,D,H,W)

    # zero-pad
    import torch.nn.functional as F
    imgs = F.pad(imgs, padding)

    return imgs


def get_input_and_mask(video_data, patient_positions, input_transform, padding):
    input_ = []
    mask_ = []

    raitio = {
        'height': 256/480,
        'width': 256/640
    }

    for idx in range(len(video_data)):
        img, pos = video_data[idx], patient_positions[idx]
        t_img = input_transform(img)   # Tensor image (C,H,W)
        input_.append(t_img)

        xmin, ymin, xmax, ymax = eval(pos)

        xmin, xmax = [round(raitio['width']*v) for v in [xmin, xmax]]
        ymin, ymax = [round(raitio['height']*v) for v in [ymin, ymax]]

        mask = torch.zeros_like(t_img)
        mask[:, ymin:ymax, xmin:xmax] = 1
        mask_.append(mask)

    input_ = process_as_tensor_image(input_, padding)
    mask_ = process_as_tensor_image(mask_, padding)

    return input_, mask_


class GAITDataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 opt,
                 input_transform=None, target_transform=None):

        self.X = X
        self.y = y
        self.vids = arrange_vids(natsorted(list(set(X.vids))), seed=0)
        self.load_pretrained = opt.load_pretrained

        self.sample_duration = opt.sample_duration

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.opt = opt

        self.feats_dir = os.path.join(
            os.path.dirname(opt.data_root), 'FeatsArrays', opt.arch)

        if target_transform:
            scaled_values = target_transform.transform(y)
            self.y.loc[:, :] = scaled_values

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

            if self.input_transform:
                self.input_transform.randomize_parameters()
                inputs = [self.input_transform(
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
                 input_transform=None, target_transform=None):

        self.X = X
        self.y = y
        self.vids = arrange_vids(natsorted(list(set(X.vids))), seed=0)
        self.load_pretrained = opt.load_pretrained

        self.sample_duration = opt.sample_duration

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.opt = opt

        if target_transform:
            scaled_values = target_transform.transform(y)
            self.y.loc[:, :] = scaled_values

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        # uniform sampling with delta(=6)
        cur_X = self.X[self.X.vids == vid].iloc[::self.opt.delta]

        frame_indices, patient_positions = cur_X.idx.values, cur_X.pos.values
        video_data = video_loader(
            self.opt.data_root, vid, frame_indices, mode='PIL')

        input_imgs, mask_imgs = get_input_and_mask(
            video_data,
            patient_positions, self.input_transform,
            padding=(
                0, 0,
                0, 0,
                0, self.sample_duration - len(frame_indices))
        )

        # target is always same!
        targets = torch.tensor(
            self.y.loc[vid2pid(vid)].values, dtype=torch.float32)

        return input_imgs, mask_imgs, targets, vid
