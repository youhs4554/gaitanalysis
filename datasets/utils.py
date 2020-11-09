from utils.transforms import (
    Compose,
    RandomResizedCrop3D,
    RandomHorizontalFlip3D,
    MultiScaleCornerCrop,
    MultiScaleCornerCrop3D,
    CenterCrop3D,
    ToTensor3D,
    Normalize3D,
    TemporalRandomCrop,
    TemporalSegmentsSampling,
    TemporalUniformSampling,
    TemporalSlidingWindow,
    LoopPadding,
    denormalize,
    VideoAugmentator,
)
import torch
from torchvision import transforms


def collate_fn_multipositon_prediction(dataset_iter):
    """
        Collation function for multi-positional (temporal) cropping,
        to guirantee max_clip_lenght among a batch (same-pad?), fill shorter frames with repeated frames
    """

    nclips_max = max([len(sample[0]) for sample in dataset_iter])

    batch = []
    for sample in dataset_iter:
        video_stack, mask_stack, label, clip_length = sample
        nclips = video_stack.size(0)

        # video_stack : (nclips,C,D,H,W)
        repeated_pts = LoopPadding(nclips_max)(torch.arange(0, nclips).tolist())[
            : nclips_max - nclips
        ]

        # merge with repeated frames
        video_stack = torch.cat((video_stack, video_stack[repeated_pts]), axis=0)
        mask_stack = torch.cat((mask_stack, mask_stack[repeated_pts]), axis=0)

        batch.append((video_stack, mask_stack, label, clip_length))

    batch_transposed = list(zip(*batch))
    for i in range(len(batch_transposed)):
        if isinstance(batch_transposed[i][0], torch.Tensor):
            batch_transposed[i] = torch.stack(batch_transposed[i], 0)

    return tuple(batch_transposed)


def get_classification_dataloaders(
    hparams, spatial_transform, temporal_transform, norm_method=None
):
    import datasets.classification

    root = hparams.data_root + ("_flow" if hparams.stream == "flow" else "")
    train_ds = datasets.classification.__dict__[hparams.dataset](
        root=root,
        annotation_path=hparams.annotation_file,
        detection_file_path=hparams.detection_file,
        sample_rate=1 if hparams.detection_file.endswith("_full.txt") else 5,
        input_size=(hparams.sample_duration, hparams.img_height, hparams.img_width),
        train=True,
        fold=hparams.fold,
        temporal_transform=temporal_transform["train"],
        spatial_transform=spatial_transform["train"],
        num_workers=hparams.num_workers,
        norm_method=norm_method,
        sample_unit="video",
    )

    val_ds = datasets.classification.__dict__[hparams.dataset](
        root=root,
        annotation_path=hparams.annotation_file,
        detection_file_path=hparams.detection_file,
        sample_rate=1 if hparams.detection_file.endswith("_full.txt") else 5,
        input_size=(hparams.sample_duration, hparams.img_height, hparams.img_width),
        train=False,
        fold=hparams.fold,
        temporal_transform=temporal_transform["val"],
        spatial_transform=spatial_transform["val"],
        num_workers=hparams.num_workers,
        norm_method=norm_method,
        sample_unit="clip",
    )

    test_ds = datasets.classification.__dict__[hparams.dataset](
        root=root,
        annotation_path=hparams.annotation_file,
        detection_file_path=hparams.detection_file,
        sample_rate=1 if hparams.detection_file.endswith("_full.txt") else 5,
        input_size=(hparams.sample_duration, hparams.img_height, hparams.img_width),
        train=False,
        hard_augmentation=False,
        fold=hparams.fold,
        temporal_transform=temporal_transform["test"],
        spatial_transform=spatial_transform["test"],
        num_workers=hparams.num_workers,
        norm_method=norm_method,
        sample_unit="video",
    )

    print(
        f"Train : {len(train_ds)}, Valid(sub-clips): {len(val_ds)}, Test(video) : {len(test_ds)}"
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=hparams.batch_per_gpu * torch.cuda.device_count(),
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=hparams.batch_per_gpu * torch.cuda.device_count(),
        shuffle=False,
        pin_memory=True,
        num_workers=hparams.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=hparams.batch_per_gpu * torch.cuda.device_count(),
        shuffle=False,
        pin_memory=True,
        num_workers=hparams.num_workers,
        collate_fn=collate_fn_multipositon_prediction,  # collate function is applied only for testing
    )

    return train_dataloader, val_dataloader, test_dataloader
