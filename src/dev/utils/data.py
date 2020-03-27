import torchvision.transforms as TF
from utils.mean import get_mean, get_std
from utils.target_columns import get_target_columns
from utils.generate_model import init_state
from utils.transforms import CenterCrop3D, Compose, MultiScaleCornerCrop, MultiScaleRandomCrop, Normalize, Normalize3D, RandomHorizontalFlip3D, RandomResizedCrop3D, Resize3D, ToTensor, ToTensor3D
import utils.visualization as viz
import os
from sklearn.preprocessing import StandardScaler
import datasets.benchmark
import datasets.gaitregression
import torch


def prepare_data(opt, fold=1):
    opt.data_root = opt.data_root.rstrip("/")
    opt.benchmark = opt.dataset in ["URFD", "MulticamFD"]

    # attention indicator
    opt.attention_str = "Attentive" if opt.attention else "NonAttentive"
    opt.group_str = f"G{opt.n_groups}" if opt.n_groups > 0 else ""
    opt.arch = "{}-{}".format(opt.backbone, opt.model_depth)

    target_columns = get_target_columns(opt)

    # define model
    net, criterion1, criterion2, optimizer, scheduler = init_state(opt)

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
        spatial_transform = {
            "train": Compose(
                [
                    Resize3D(size=opt.img_size),
                    RandomResizedCrop3D(
                        transform2D=TF.RandomResizedCrop(
                            size=(opt.sample_size, opt.sample_size))),
                    # scale=(opt.sample_size / opt.img_size, 1.0))),
                    RandomHorizontalFlip3D(),
                    ToTensor3D(),
                    Normalize3D(
                        mean=opt.mean,
                        std=opt.std
                    ),
                ]
            ),
            "test": Compose(
                [
                    Resize3D(size=opt.img_size),
                    CenterCrop3D((opt.sample_size, opt.sample_size)),
                    ToTensor3D(),
                    Normalize3D(
                        mean=opt.mean,
                        std=opt.std
                    ),
                ]
            ),
        }

        temporal_transform = {
            "train": None,  # TemporalRandomCrop(opt.sample_duration),
            "test": None,  # TemporalCenterCrop(opt.sample_duration),
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
            # step_between_clips=10,
            fold=fold,
            transform=spatial_transform["train"],
            preCrop=opt.precrop,
        )

        test_ds = datasets.benchmark.FallDataset(
            root=opt.data_root,
            train=False,
            annotation_path=os.path.join(
                os.path.dirname(opt.data_root), "TrainTestlist"
            ),
            detection_file_path=opt.input_file,
            frames_per_clip=opt.sample_duration,
            # step_between_clips=10,
            fold=fold,
            transform=spatial_transform["test"],
            preCrop=opt.precrop,
        )

        # for i in range(len(test_ds)):
        #     test_ds[i]

    else:
        NotImplementedError("Does not support other datasets until now..")

    # Construct train/test dataloader for selected fold
    train_loader = torch.utils.data.DataLoader(train_ds, pin_memory=True,
                                               batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.n_threads)
    test_loader = torch.utils.data.DataLoader(test_ds, pin_memory=True,
                                              batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.n_threads)

    return (
        opt,
        net,
        criterion1,
        criterion2,
        optimizer,
        scheduler,
        spatial_transform,
        temporal_transform,
        target_transform,
        plotter,
        train_loader,
        test_loader,
        target_columns,
    )
