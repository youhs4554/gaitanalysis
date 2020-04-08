import torchvision.transforms as TF
from utils.mean import get_mean, get_std
from utils.target_columns import get_target_columns
from utils.generate_model import init_state
from utils.transforms import (
    CenterCrop3D,
    Compose,
    MultiScaleCornerCrop,
    MultiScaleRandomCrop,
    Normalize,
    Normalize3D, RandomHorizontalFlip3D, RandomResizedCrop3D, RandomCrop3D, Resize3D, ToTensor, ToTensor3D, LoopPadding, TemporalRandomCrop, LoopTemporalCrop
)

import utils.visualization as viz
import os
from sklearn.preprocessing import StandardScaler
import datasets.benchmark
import datasets.gaitregression
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import collections


def get_balanced_sampler(train_ds, num_workers=8):
    class _DummyDataset(torch.utils.data.Dataset):
        def __init__(self, x):
            self.x = x

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index]

    dl = torch.utils.data.DataLoader(
        _DummyDataset(train_ds),
        batch_size=16,
        num_workers=num_workers)

    cls_counts = collections.defaultdict(int)
    cls_samples = []
    N = len(train_ds)
    with tqdm(total=len(dl), desc='class-balancing...') as pbar:
        for batch in dl:
            pbar.update(1)
            labs = batch[2].numpy().tolist()
            for lab in labs:
                cls_counts[lab] += 1
            cls_samples += labs

    sample_weights = [0] * N
    for i in range(N):
        sample_weights[i] = N / cls_counts[cls_samples[i]]

    return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, N, replacement=True)


def prepare_data(opt, fold=1):
    opt.data_root = opt.data_root.rstrip("/")
    opt.benchmark = opt.dataset in ["URFD", "MulticamFD"]

    # attention indicator
    opt.attention_str = "Attentive" if opt.attention else "NonAttentive"
    opt.group_str = f"G{opt.n_groups}" if opt.n_groups > 0 else ""
    opt.arch = "{}-{}".format(opt.backbone, opt.model_depth)

    target_columns = get_target_columns(opt)

    # # T_max : for cosine shaped `lr_anealing`
    # opt.T_max = opt.n_iter

    # # define model
    # net, criterion1, criterion2, optimizer, lr_scheduler, warmup_scheduler = init_state(
    #     opt)

    # import visdom
    # viz = visdom.Visdom('155.230.214.70')

    # win = None
    # global_step = 0
    # net.train()
    # for epoch in range(1, 21):
    #     for i in range(707):
    #         if epoch == 1 and i == 0:
    #             update = None
    #         else:
    #             update = 'append'

    #         lr_scheduler.step(epoch-1)
    #         warmup_scheduler.dampen()

    #         lr = optimizer.param_groups[0]['lr']

    #         if i % 100 == 0:
    #             win = viz.line(X=[global_step],
    #                            Y=[lr], win=win, update=update)
    #         # update params
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         global_step += 1

    #     # lr_scheduler.step()

    # import ipdb
    # ipdb.set_trace()

    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value, dataset=opt.mean_dataset)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if opt.train_crop == "random":
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == "corner":
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == "center":
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=["c"]
        )

    model_indicator = "_".join(
        filter(
            lambda x: x != "",
            [
                opt.attention_str,
                opt.model_arch,
                opt.merge_type,
                opt.arch,
                opt.group_str,
            ],
        )
    )
    opt.model_indicator = model_indicator

    plotter = viz.VisdomPlotter(
        env_name=opt.dataset + '_' +
        opt.model_indicator + "-fold-{}".format(fold)
    )

    # default sampler for trainData : random
    sampler = None

    if opt.dataset == "Gaitparams_PD":
        spatial_transform = {
            "train": Compose(
                [
                    TF.RandomRotation(degrees=(0, 0)),
                    TF.RandomResizedCrop(
                        size=opt.sample_size,
                        scale=(opt.sample_size / opt.img_size, 1.0),
                    ),
                    ToTensor(opt.norm_value),
                    norm_method,
                ]
            ),
            "test": Compose(
                [TF.CenterCrop(opt.sample_size), ToTensor(
                    opt.norm_value), norm_method]
            ),
        }

        temporal_transform = {
            "train": None,  # TemporalRandomCrop(opt.sample_duration),
            "test": None,  # TemporalCenterCrop(opt.sample_duration),
        }

        target_transform = StandardScaler()

        # prepare dataset  (train/test split)
        data = datasets.gaitregression.prepare_dataset(
            input_file=opt.input_file,
            target_file=opt.target_file,
            target_columns=target_columns,
            chunk_parts=opt.chunk_parts,
            target_transform=target_transform,
        )

        if opt.with_segmentation:
            ds_class = datasets.gaitregression.GAITSegRegDataset
        else:
            ds_class = datasets.gaitregression.GAITDataset

        train_ds = ds_class(
            X=data["train_X"],
            y=data["train_y"],
            opt=opt,
            phase="train",
            spatial_transform=spatial_transform["train"],
            temporal_transform=temporal_transform["train"],
        )

        test_ds = ds_class(
            X=data["test_X"],
            y=data["test_y"],
            opt=opt,
            phase="test",
            spatial_transform=spatial_transform["test"],
            temporal_transform=temporal_transform["test"],
        )

    elif opt.benchmark:
        norm_method = Normalize3D(
            mean=opt.mean,
            std=opt.std
        )
        spatial_transform = {
            "train": Compose(
                [
                    RandomCrop3D(transform2D=TF.RandomCrop(
                        size=(opt.sample_size, opt.sample_size))
                    ),
                    RandomHorizontalFlip3D(),
                    ToTensor3D()
                ]
            ),
            "test": Compose(
                [
                    CenterCrop3D((opt.sample_size, opt.sample_size)),
                    LoopTemporalCrop(step_between_clips=opt.sample_duration),
                    TF.Lambda(lambda clips:
                              torch.stack([ToTensor3D()(clip) for clip in clips]))
                ]
            ),
        }

        temporal_transform = {
            "train": TemporalRandomCrop(opt.sample_duration),
            "test": None
        }

        target_transform = None

        train_ds = datasets.benchmark.FallDataset(
            root=opt.data_root,
            train=True,
            annotation_path=os.path.join(
                os.path.dirname(opt.data_root), "TrainTestlist"
            ),
            detection_file_path=opt.input_file,
            frames_per_clip=opt.sample_duration,
            # step_between_clips=opt.sample_duration // 2,
            fold=fold,
            spatial_transform=spatial_transform["train"],
            temporal_transform=temporal_transform["train"],
            norm_method=norm_method,
            preCrop=opt.precrop,
            img_size=opt.img_size
        )

        test_ds = datasets.benchmark.FallDataset(
            root=opt.data_root,
            train=False,
            annotation_path=os.path.join(
                os.path.dirname(opt.data_root), "TrainTestlist"
            ),
            detection_file_path=opt.input_file,
            frames_per_clip=opt.sample_duration,
            step_between_clips=opt.sample_duration,
            fold=fold,
            spatial_transform=spatial_transform["test"],
            temporal_transform=temporal_transform["test"],
            norm_method=norm_method,
            preCrop=opt.precrop,
            img_size=opt.img_size
        )

        # initial shuffle for test dataset
        torch.manual_seed(0)    # for reproductivity
        indices = torch.randperm(len(test_ds)).tolist()
        test_ds = torch.utils.data.Subset(test_ds, indices)

        train_ds[0]
        test_ds[50]

        # for balanced sampleing w.r.t class
        sampler = get_balanced_sampler(train_ds, num_workers=opt.n_threads)

    else:
        NotImplementedError("Does not support other datasets until now..")

    print(f'train : {len(train_ds)}, test : {len(test_ds)}')
    print()

    # Construct train/test dataloader for selected fold
    train_loader = torch.utils.data.DataLoader(train_ds, pin_memory=True,
                                               batch_size=opt.batch_size, shuffle=(
                                                   sampler is None),
                                               sampler=sampler,
                                               num_workers=opt.n_threads)
    test_loader = torch.utils.data.DataLoader(test_ds, pin_memory=True,
                                              batch_size=1, shuffle=False,
                                              num_workers=opt.n_threads)

    # T_max : for cosine shaped `lr_anealing`
    opt.T_max = opt.n_iter
    # warmup_period
    # warmup learning rate for 3 epochs
    opt.warmup_period = len(train_loader) * 3

    # define model
    net, criterion1, criterion2, optimizer, lr_scheduler, warmup_scheduler = init_state(
        opt)

    return (
        opt,
        net,
        criterion1,
        criterion2,
        optimizer,
        lr_scheduler,
        warmup_scheduler,
        spatial_transform,
        temporal_transform,
        target_transform,
        plotter,
        train_loader,
        test_loader,
        target_columns,
    )
