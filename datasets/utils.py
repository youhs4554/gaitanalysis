import torch.nn.functional as F
from .regression.gait import prepare_dataset as prepare_gait_dataset
from .regression.gait import GAITDataset, GAITSegRegDataset

import torchvision.transforms as TF
from utils.transforms import (
    CenterCrop3D,
    Compose,
    Normalize,
    Normalize3D,
    RandomHorizontalFlip3D, RandomRotation3D,
    RandomCrop3D, ToTensor, ToTensor3D, TemporalRandomCrop,
    LoopPadding, LoopTemporalCrop, TemporalCenterCrop, RandomResizedCrop3D
)
from sklearn.preprocessing import StandardScaler
import torch
import collections
import tqdm
import numpy as np
import os
import manifest.target_columns
import random
import math
import itertools
import datasets.classification.falldown.datasets as falldown_datasets


__all__ = [
    "get_data_loader",
    "get_balanced_sampler",
]


def collate_fn_multipositon_prediction(dataset_iter):
    """
        Collation for multi-positional cropping validation, to guirantee max_clip_lenght among a batch (zero-pad?)
        fill shorter frames with all-zero frames, thus, the default label is '0'
    """

    nclips_max = max([len(sample[0]) for sample in dataset_iter])

    batch = []
    for sample in dataset_iter:
        clip_seq, mask_seq, target = sample
        nclips = clip_seq.size(0)
        pad = (0, 0,)*(clip_seq.dim()-1) + \
            (0, nclips_max-nclips)

        # zero-pad
        clip_seq = F.pad(clip_seq, pad)
        mask_seq = F.pad(mask_seq, pad)
        clip_level_target = F.pad(target.repeat(
            nclips), (0, nclips_max-nclips))
        video_level_target = target

        batch.append(
            (clip_seq, mask_seq, clip_level_target,
                video_level_target)
        )

    batch_transposed = list(zip(*batch))
    for i in range(len(batch_transposed)):
        if isinstance(batch_transposed[i][0], torch.Tensor):
            batch_transposed[i] = torch.stack(batch_transposed[i], 0)

    # images
    for i in range(2):
        bs, nclips, *cdhw = batch_transposed[i].shape
        batch_transposed[i] = batch_transposed[i].view(-1, *cdhw)

    # clip_targets (repeated)
    batch_transposed[2] = batch_transposed[2].flatten()

    return tuple(batch_transposed)


def get_balanced_sampler(ds, batch_size, num_workers=8):
    class _DummyDataset(torch.utils.data.Dataset):
        def __init__(self, x):
            self.x = x

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index]

    dl = torch.utils.data.DataLoader(
        _DummyDataset(ds),
        batch_size=16,
        # collate_fn=collate_fn_zeropadding,
        num_workers=num_workers)

    cls_counts = collections.defaultdict(int)
    cls_samples = []
    with tqdm.tqdm(total=len(dl), desc='class-balancing...') as pbar:
        for batch in dl:
            pbar.update(1)
            labs = batch[2].numpy().tolist()
            for lab in labs:
                cls_counts[lab] += 1
            cls_samples += labs

    def get_sentiniles(cls_samples):
        cls_samples = np.array(cls_samples)

        # sentiniles to prevent highly unvalanced batch
        sentinile = random.choice(range(len(cls_samples)))
        pos_sentinile = random.choice(
            np.where(cls_samples == cls_samples[sentinile])[0])
        neg_sentinile = random.choice(
            np.where(cls_samples != cls_samples[sentinile])[0])

        return sentinile, pos_sentinile, neg_sentinile

    N = len(cls_samples)
    sample_weights = [0] * N
    for i in range(N):
        sample_weights[i] = N / cls_counts[cls_samples[i]]

    sample_indices = []
    for _ in range(math.ceil(N/batch_size)):
        sentinile_indices = list(get_sentiniles(cls_samples))
        random_indices = list(torch.utils.data.sampler.WeightedRandomSampler(
            sample_weights, batch_size-3, replacement=True))
        for ri in random_indices:
            sample_weights[ri] = 0.0

        sample_indices += sentinile_indices + random_indices
    sample_indices = sample_indices[:len(ds)]

    # sample_indices = list(
    #     itertools.chain(*[list(get_sentiniles(cls_samples)) + list(
    #         torch.utils.data.sampler.WeightedRandomSampler(sample_weights, batch_size-3, replacement=True)) for _ in range(math.ceil(N/batch_size))]
    #     ))[:len(ds)]

    class SubsetSequentialSampler(torch.utils.data.sampler.Sampler):
        r"""Samples elements randomly from a given list of indices, without replacement.

        Arguments:
            indices (sequence): a sequence of indices
        """

        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return (self.indices[i] for i in range(len(self.indices)))

        def __len__(self):
            return len(self.indices)

    return SubsetSequentialSampler(sample_indices)


def get_gait_dataset(opt, fold):
    spatial_transform = {
        "train": Compose(
            [
                TF.RandomRotation(degrees=(0, 0)),
                TF.RandomResizedCrop(
                    size=opt.sample_size,
                    scale=(opt.sample_size / opt.img_size, 1.0),
                ),
                ToTensor(opt.norm_value),
                Normalize(opt.mean, opt.std),
            ]
        ),
        "test": Compose(
            [TF.CenterCrop(opt.sample_size), ToTensor(
                opt.norm_value), Normalize(opt.mean, opt.std)]
        ),
    }

    temporal_transform = {
        "train": None,
        "test": None,
    }

    target_transform = StandardScaler()

    target_columns = manifest.target_columns.BASIC_GAIT_PARAMS
    if opt.target_columns == 'advanced':
        target_columns = manifest.target_columns.ADVANCED_GAIT_PARAMS

    # prepare dataset  (train/test split)
    data = prepare_gait_dataset(
        input_file=opt.detection_file,
        target_file=opt.target_file,
        target_columns=target_columns,
        chunk_parts=opt.chunk_parts,
        target_transform=target_transform,
    )

    if opt.with_segmentation:
        ds_class = GAITSegRegDataset
    else:
        ds_class = GAITDataset

    train_ds = ds_class(
        X=data["train_X"],
        y=data["train_y"],
        opt=opt,
        phase="train", fold=fold,
        spatial_transform=spatial_transform['train'],
        temporal_transform=temporal_transform['train']
    )

    test_ds = ds_class(
        X=data["test_X"],
        y=data["test_y"],
        opt=opt,
        phase="test", fold=fold,
        spatial_transform=spatial_transform['test'],
        temporal_transform=temporal_transform['test']
    )

    return train_ds, test_ds, target_transform, len(target_columns)


def collate_fn_zeropadding(dataset_iter):
    """
        Collation for zeropadding for same-length inputs
        fill shorter frames with all-zero frames
    """

    maxlen = max([sample[0].size(1) for sample in dataset_iter])

    batch = []
    for sample in dataset_iter:
        video, masks, label, video_name, valid_len = sample
        nframes = video.size(1)
        pad = (0, 0,)*(video.dim()-2) + \
            (0, maxlen-nframes)

        # zero-pad
        video = F.pad(video, pad)
        masks = F.pad(masks, pad)

        batch.append(
            (video, masks, label, video_name, valid_len)
        )

    batch_transposed = list(zip(*batch))
    for i in range(len(batch_transposed)):
        if isinstance(batch_transposed[i][0], torch.Tensor):
            batch_transposed[i] = torch.stack(batch_transposed[i], 0)

    return tuple(batch_transposed)


def get_data_loader(opt, fold):
    sampler = None
    collate_fn = None
    if opt.dataset == "Gaitparams_PD":
        train_ds, test_ds, target_transform, n_outputs = get_gait_dataset(
            opt, fold=fold)
    else:
        dataset = getattr(falldown_datasets, opt.dataset)(dataset_root=opt.data_root,
                                                          fold=fold,
                                                          sample_length=opt.sample_duration,
                                                          batch_size=opt.batch_size, tsn=False)
        train_ds = dataset.train_ds
        test_ds = dataset.test_ds
        # # train : TSN sampling, test : sliding-window sampling
        # train_ds.dataset.num_samples = 3
        # test_ds.dataset.num_samples = 1
        target_transform = None
        n_outputs = 2

        if opt.multiple_clip:
            collate_fn = collate_fn_multipositon_prediction
        if opt.balanced_batch:
            sampler = get_balanced_sampler(
                train_ds, batch_size=opt.batch_size) if opt.balanced_batch else None

    if sampler is None:
        sampler = torch.utils.data.sampler.RandomSampler(train_ds)

    # Construct train/test dataloader for selected fold
    train_loader = torch.utils.data.DataLoader(train_ds, pin_memory=True,
                                               batch_size=opt.batch_size * torch.cuda.device_count(),
                                               sampler=sampler,
                                               # collate_fn=collate_fn_zeropadding,
                                               num_workers=opt.n_threads)
    test_loader = torch.utils.data.DataLoader(test_ds, pin_memory=True,
                                              batch_size=opt.batch_size * torch.cuda.device_count(), shuffle=False,
                                              collate_fn=collate_fn,
                                              num_workers=opt.n_threads)

    print(
        f'train : {len(train_ds)}, test : {len(test_ds)}')
    print()

    return train_loader, test_loader, target_transform, n_outputs
