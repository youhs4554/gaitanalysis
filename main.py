import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from torchvision import transforms

from datasets.utils import get_classification_dataloaders
from models.video_classifier import LightningVideoClassifier
from utils.callbacks import CustomTensorBoard_Logger
from utils.transforms import (
    Compose,
    RandomHorizontalFlip3D,
    MultiScaleCornerCrop,
    MultiScaleCornerCrop3D,
    CenterCrop3D,
    ToTensor3D,
    Normalize3D,
    TemporalRandomCrop,
    TemporalSlidingWindow,
)

if __name__ == "__main__":

    import warnings

    warnings.filterwarnings("ignore")

    torch.backends.cudnn.benchmark = True

    # for reproductivity
    pl.seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_arch",
        type=str,
        default="STCNet",
        help="STCNet | FineTunedConvNet | ConcatenatedSTCNet",
    )
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument(
        "--task", type=str, default="classification", help="classification | regression"
    )
    parser.add_argument("--backbone", type=str, default="r2plus1d_34_32_kinetics")
    parser.add_argument(
        "--dataset", type=str, help="name of dataset (UCF101|HMDB51|GAIT|CesleaFDD6...)"
    )
    parser.add_argument(
        "--stream", type=str, help="name of dataset (rgb|flow)", default="rgb"
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch_per_gpu", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=112)
    parser.add_argument("--sample_duration", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--mask_supervision", action="store_true")
    parser.add_argument("--squad", type=str)
    hparams = parser.parse_args()

    if "r2plus1d" in hparams.backbone:
        # MEAN & STD for kinetics datasets
        MEAN = [0.43216, 0.394666, 0.37645]
        STD = [0.22803, 0.22145, 0.216989]
    else:
        # imagenet
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    scale_ratio = 2 if "inflated" in hparams.backbone else 1

    DATASET_CONFIG = {
        "UCF101": {
            "detection_file": "/data/torch_data/UCF-101/detection_rcnn_full.txt",
            "annotation_file": "/data/torch_data/UCF-101/ucfTrainTestlist",
            "data_root": "/data/torch_data/UCF-101/video",
            "img_height": 128 * scale_ratio,
            "img_width": 170 * scale_ratio,
        },
        "HMDB51": {
            "detection_file": "/data/torch_data/HMDB51/detection_rcnn_full.txt",
            "annotation_file": "/data/torch_data/HMDB51/testTrainMulti_7030_splits",
            "data_root": "/data/torch_data/HMDB51/video",
            "img_height": 128 * scale_ratio,
            "img_width": 170 * scale_ratio,
        },
        "CesleaFDD6": {
            "detection_file": "/data/torch_data/hosp_fall_dataset/detection_yolo_full.txt",
            "annotation_file": "/data/torch_data/hosp_fall_dataset/fallTrainTestlist",
            "data_root": "/data/torch_data/hosp_fall_dataset/video",
            "img_height": 128 * scale_ratio,
            "img_width": 170 * scale_ratio,
        },
    }

    # update dataset config (detection_file, annotation_file, data_root)
    vars(hparams).update(DATASET_CONFIG.get(hparams.dataset))

    # transforms
    center_crop = Compose(
        [CenterCrop3D((hparams.sample_size, hparams.sample_size)), ToTensor3D()]
    )
    spatial_transform = {
        "train": Compose(
            [
                MultiScaleCornerCrop3D(
                    transform2D=MultiScaleCornerCrop(
                        scales=[1.0, 0.875, 0.75, 0.66], size=hparams.sample_size
                    )
                ),
                RandomHorizontalFlip3D(),
                ToTensor3D(),
            ]
        ),
        # val/test : use torchvision transforms
        "val": center_crop,
        "test": Compose(
            [
                transforms.Lambda(
                    lambda clips: torch.stack([center_crop(clip) for clip in clips])
                )
            ]
        ),
    }

    temporal_transform = {
        "train": TemporalRandomCrop(size=hparams.sample_duration),
        "val": None,
        "test": TemporalSlidingWindow(size=hparams.sample_duration),
    }

    norm_method = Normalize3D(mean=MEAN, std=STD)

    # dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_classification_dataloaders(
        hparams, spatial_transform, temporal_transform, norm_method
    )

    model = LightningVideoClassifier(hparams)

    lr_logger = LearningRateLogger()

    # init logger
    base_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs_tmp/TRAIN",
        name=f"{hparams.model_arch}_{hparams.sample_duration}f_{hparams.dataset}-{hparams.fold}",
    )
    tb_logger = CustomTensorBoard_Logger(base_logger)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        distributed_backend="dp",
        callbacks=[tb_logger, lr_logger],
        max_epochs=50,
        checkpoint_callback=checkpoint_callback,
        logger=base_logger,
        log_save_interval=10,
        num_sanity_val_steps=0,
        weights_summary="top",
    )
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model, test_dataloaders=test_dataloader)
