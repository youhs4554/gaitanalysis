import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import argparse
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datasets.classification.ucf101 import UCF101
from datasets.classification.hmdb51 import HMDB51
from sklearn.metrics.classification import confusion_matrix

from models import generate_network

from utils.transforms import (
    Compose,
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
from utils.callbacks import *

import sklearn.metrics
import os, glob
import json
import argparse
import numpy as np
import random
import warnings
from ranger import Ranger  # this is from ranger.py
from collections import namedtuple, defaultdict
import pandas as pd
from ranger import Ranger  # this is from ranger.py
from functools import reduce
import math
import torch_optimizer
import cv2
import albumentations as A
import albumentations.pytorch.transforms
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings("ignore")


def mixup_data(video, mask, label, clip_length, alpha=0.5):
    lambda_ = np.random.beta(alpha, alpha)

    batch_size = video.size(0)
    indices = torch.randperm(batch_size)

    mixed_video = lambda_ * video + (1 - lambda_) * video[indices]
    mixed_mask = lambda_ * mask + (1 - lambda_) * mask[indices]

    label_a, label_b = label, label[indices]
    return mixed_video, mixed_mask, label_a, label_b, lambda_


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


def set_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LightningVideoClassifier(pl.LightningModule):
    datasets_map = {"UCF101": UCF101, "HMDB51": HMDB51}

    def __init__(self, hparams):
        super(LightningVideoClassifier, self).__init__()

        self.hparams = hparams
        name = hparams.dataset  # name of dataset

        if name not in ["UCF101", "HMDB51"]:
            raise ValueError(
                "Unsupported Dataset. This class only supports ( UCF101 | HMDB51 )"
            )
        if hparams.task == "classification":
            n_outputs = int("".join([c for c in name if c.isdigit()]))
        elif hparams.task == "regression":
            # if hparams.dataset == "GAIT":
            from cfg.target_columns import BASIC_GAIT_PARAMS, ADVANCED_GAIT_PARAMS

            n_outputs = len(BASIC_GAIT_PARAMS)
            if hparams.model_arch == "ConcatenatedSTCNet":
                n_outputs = len(ADVANCED_GAIT_PARAMS)

        self.dataset_init_func = self.datasets_map.get(name)
        self.model = generate_network(hparams, n_outputs=n_outputs)

    def forward(self, *batch):
        video, mask, label, lambda_ = batch
        out, loss_dict, tb_dict = self.model(
            video, mask, targets=label, lambda_=lambda_
        )
        if (
            out.device == torch.device(0)
            and self.training
            and self.trainer.global_step % 50 == 0
        ):
            v = video[0].permute(1, 2, 3, 0)
            v = denormalize(v, MEAN, STD)

            m = mask[0].permute(1, 2, 3, 0).repeat(1, 1, 1, 3)
            ov = v * m.gt(0.0).float()
            i = torch.cat((v, m, ov), 0)
            self.logger.experiment.add_images(
                "clip_batch_image_and_mask",
                i,
                self.trainer.global_step,
                dataformats="NHWC",
            )

            if tb_dict:
                from torchvision.utils import make_grid

                for tag, tensor in tb_dict.items():
                    grid_img = make_grid(tensor, pad_value=1)
                    self.logger.experiment.add_image(
                        tag, grid_img, self.trainer.global_step
                    )

        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}

        return out, loss_dict

    def step(self, batch, batch_idx, mixup=False):
        video, mask, label, clip_length = batch
        lambda_ = None
        if mixup:
            video, mask, *label, lambda_ = mixup_data(*batch, alpha=0.5)
        if self.trainer.testing:
            n_clips = video.size(1)
            clip_indices = (torch.tensor(clip_length) - 1).tolist()
            clip_indices = [
                list(range(x + 1)) + [0] * (n_clips - x) for x in clip_indices
            ]
            indice_mask = torch.zeros(video.size(0), n_clips).scatter_(
                1, torch.tensor(clip_indices), 1.0
            )

            out = []
            loss_dict = defaultdict(float)
            for n in range(n_clips):
                cout, closs_dict = self.forward(video[:, n], mask[:, n], label, lambda_)
                out.append(cout)
                for key in closs_dict:
                    loss_dict[key] += closs_dict[key] / n_clips
            # default_dict -> dict
            loss_dict = dict(loss_dict)
            out = torch.stack(out)
            out = out * (indice_mask.t().unsqueeze(2)).to(video.device)

            # temporal avg pool
            out = out.sum(0) / indice_mask.sum(1, keepdim=True).to(video.device)

            # temporal max pool
            # out, _ = out.max(0)

        else:
            out, loss_dict = self.forward(video, mask, label, lambda_)

        predicted = out.argmax(1)

        loss = sum(loss for loss in loss_dict.values())
        if mixup:
            label_a, label_b = label
            total = out.size(0)
            correct = (
                lambda_ * predicted.eq(label_a).float().sum()
                + (1 - lambda_) * predicted.eq(label_b).float().sum()
            )
        else:
            total = out.size(0)
            correct = (predicted == label).float().sum()

        if self.trainer.testing:
            # compute top-5 accuracy in test mode
            _, pred = torch.topk(out, k=5)
            correct_top5 = pred.eq(label.view(-1, 1)).any(1).float().sum()

            return {
                "loss": loss,
                "acc": correct / total,
                "top5_acc": correct_top5 / total,
                "loss_dict": loss_dict,
                "pred": out.argmax(1).float(),
                "label": label.float(),
            }
        else:
            return {"loss": loss, "acc": correct / total, "loss_dict": loss_dict}

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, mixup=self.hparams.mixup)

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx)

    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, batch_idx)  # for averaged logits

    def validation_epoch_end(self, outputs):
        """
            called at the end of the validation epoch
            outputs is an array with what you returned in validation_step for each batch
            outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        """
        avg_loss = torch.stack([x["loss"].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"].mean() for x in outputs]).mean()

        return {"val_loss": avg_loss, "val_acc": avg_acc}

    def test_epoch_end(self, outputs):
        """
            called at the end of the testing loop
            outputs is an array with what you returned in validation_step for each batch
            outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        """
        avg_loss = torch.stack([x["loss"].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"].mean() for x in outputs]).mean()

        y_pred = (
            torch.cat([x["pred"].flatten(0) for x in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        y_true = (
            torch.cat([x["label"].flatten(0) for x in outputs], dim=0).cpu().numpy()
        )
        cm = confusion_matrix(y_true, y_pred)

        # Normalise
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        classes = self.train_ds.classes
        df_cm = pd.DataFrame(cm, classes, classes)
        df_cm.to_csv(f"{self.hparams.dataset}_results.csv", index=False)

        avg_top5_acc = torch.stack([x["top5_acc"].mean() for x in outputs]).mean()

        return {
            "test_loss": avg_loss,
            "test_acc": avg_acc,
            "test_top5_acc": avg_top5_acc,
        }

    def prepare_data(self):
        center_crop = Compose(
            [
                CenterCrop3D((self.hparams.sample_size, self.hparams.sample_size)),
                ToTensor3D(),
            ]
        )
        crop_method = (
            A.CenterCrop if self.hparams.dataset == "UCF101" else A.RandomResizedCrop
        )
        spatial_transform = {
            # train : use albumentations
            "train": VideoAugmentator(
                transform=A.Compose(
                    [
                        crop_method(self.hparams.sample_size, self.hparams.sample_size),
                        A.HorizontalFlip(),
                        A.pytorch.transforms.ToTensor(),
                    ]
                )
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
            "train": TemporalRandomCrop(size=self.hparams.sample_duration),
            # "train": TemporalSegmentsSampling(
            #     n_seg=self.hparams.sample_duration
            # ),  # TSN sampling for long-range temporal context
            "val": None,
            "test": TemporalSlidingWindow(size=self.hparams.sample_duration),
            # "test": TemporalUniformSampling(n_chunks=10, size=self.hparams.sample_duration)
        }

        norm_method = Normalize3D(mean=MEAN, std=STD)

        root = self.hparams.data_root + (
            "_flow" if self.hparams.stream == "flow" else ""
        )
        self.train_ds = self.dataset_init_func(
            root=root,
            annotation_path=self.hparams.annotation_file,
            detection_file_path=self.hparams.detection_file,
            sample_rate=1 if self.hparams.detection_file.endswith("_full.txt") else 5,
            input_size=(
                self.hparams.sample_duration,
                self.hparams.img_height,
                self.hparams.img_width,
            ),
            train=True,
            hard_augmentation=True,
            fold=self.hparams.fold,
            temporal_transform=temporal_transform["train"],
            spatial_transform=spatial_transform["train"],
            num_workers=self.hparams.num_workers,
            norm_method=norm_method,
            sample_unit="video",
        )

        self.val_ds = self.dataset_init_func(
            root=root,
            annotation_path=self.hparams.annotation_file,
            detection_file_path=self.hparams.detection_file,
            sample_rate=1 if self.hparams.detection_file.endswith("_full.txt") else 5,
            input_size=(
                self.hparams.sample_duration,
                self.hparams.img_height,
                self.hparams.img_width,
            ),
            train=False,
            hard_augmentation=False,
            fold=self.hparams.fold,
            temporal_transform=temporal_transform["val"],
            spatial_transform=spatial_transform["val"],
            num_workers=self.hparams.num_workers,
            norm_method=norm_method,
            sample_unit="clip",
        )

        self.test_ds = self.dataset_init_func(
            root=root,
            annotation_path=self.hparams.annotation_file,
            detection_file_path=self.hparams.detection_file,
            sample_rate=1 if self.hparams.detection_file.endswith("_full.txt") else 5,
            input_size=(
                self.hparams.sample_duration,
                self.hparams.img_height,
                self.hparams.img_width,
            ),
            train=False,
            hard_augmentation=False,
            fold=self.hparams.fold,
            temporal_transform=temporal_transform["test"],
            spatial_transform=spatial_transform["test"],
            num_workers=self.hparams.num_workers,
            norm_method=norm_method,
            sample_unit="video",
        )

        def shuffle_ds(ds):
            indices = torch.randperm(len(ds))
            ds = torch.utils.data.Subset(ds, indices)
            return ds

        # initial shuffle for validation & testset
        self.val_ds = shuffle_ds(self.val_ds)
        self.test_ds = shuffle_ds(self.test_ds)
        # from tqdm import tqdm

        # for i in tqdm(range(len(self.val_ds))):
        #     self.val_ds[i]
        # import ipdb

        # ipdb.set_trace()

        print(
            f"Train : {len(self.train_ds)}, Valid(sub-clips): {len(self.val_ds)}, Test(video) : {len(self.test_ds)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_per_gpu * self.trainer.gpus,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_per_gpu * self.trainer.gpus,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_per_gpu * self.trainer.gpus,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn_multipositon_prediction,  # collate function is applied only for testing
        )

    # learning rate warm-up
    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        using_native_amp=None,
    ):
        # use lr proposed by lr_finder
        lr = self.hparams.learning_rate or self.lr
        until = 5 * len(self.train_dataloader())  # warm-up for 5 epochs
        # warm up lr
        if self.trainer.global_step < until:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / until)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * lr
        else:
            # self.schedulers[optimizer_idx].step(epoch=current_epoch - 5)
            self.schedulers[optimizer_idx].step()

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=(self.hparams.learning_rate or self.lr),
            momentum=0.9,
            nesterov=True,
            weight_decay=self.hparams.weight_decay,
        )

        self.schedulers = [
            # torch.optim.lr_scheduler.LambdaLR(
            #     self.optimizer, lambda epoch: (0.94) ** ((epoch + 1) // 2)
            # )
            # torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloader()) * self.trainer.max_epochs,
                eta_min=self.hparams.learning_rate * (1 / 16),
            )
        ]

        return self.optimizer


if __name__ == "__main__":
    DATASET_CONFIG = {
        "UCF101": {
            "detection_file": "/data/torch_data/UCF-101/detection_rcnn_full.txt",
            "annotation_file": "/data/torch_data/UCF-101/ucfTrainTestlist",
            "data_root": "/data/torch_data/UCF-101/video",
            "img_height": 112,
            "img_width": 112,
        },
        "HMDB51": {
            "detection_file": "/data/torch_data/HMDB51/detection_rcnn_full.txt",
            "annotation_file": "/data/torch_data/HMDB51/testTrainMulti_7030_splits",
            "data_root": "/data/torch_data/HMDB51/video",
            "img_height": 128,
            "img_width": 171,
        },
    }

    # MEAN & STD for kinetics datasets
    MEAN = [0.43216, 0.394666, 0.37645]
    STD = [0.22803, 0.22145, 0.216989]

    torch.backends.cudnn.benchmark = True
    # for reproductivity
    set_seed(0)

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
        "--dataset", type=str, help="name of dataset (UCF101|HMDB51|GAIT|...)"
    )
    parser.add_argument(
        "--stream", type=str, help="name of dataset (rgb|flow)", default="rgb"
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--batch_per_gpu", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=112)
    parser.add_argument("--sample_duration", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-3)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--squad", type=str)
    hparams = parser.parse_args()

    # update dataset config (detection_file, annotation_file, data_root)
    vars(hparams).update(DATASET_CONFIG.get(hparams.dataset))
    model = LightningVideoClassifier(hparams)

    lr_logger = LearningRateLogger()

    # init logger
    base_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs_tmp/TRAIN",
        name=f"{hparams.model_arch}_{hparams.backbone}_duration={hparams.sample_duration}_mixup={hparams.mixup}_{hparams.dataset}_stream={hparams.stream}_squad={hparams.squad}@fold-{hparams.fold}",
        # version=f"{hparams.model_arch}_{hparams.backbone}_duration={hparams.sample_duration}_mixup={hparams.mixup}_{hparams.dataset}_stream={hparams.stream}@fold-{hparams.fold}",
    )
    tb_logger = TensorBoard_Logger(base_logger)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    es_callback = EarlyStopping(
        monitor="val_acc", patience=10, verbose=True, mode="max"
    )
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        distributed_backend="dp",
        callbacks=[tb_logger],
        # early_stop_callback=es_callback,
        max_epochs=50,
        checkpoint_callback=checkpoint_callback,
        logger=base_logger,
        log_save_interval=10,
        num_sanity_val_steps=0,
        gradient_clip_val=5.0,
        weights_summary="top",
    )

    @torch.no_grad()
    def run_test(model):
        # load best weights
        ckpt = glob.glob(base_logger.log_dir + "/checkpoints/*.ckpt")[0]
        print(f"Load weights from {ckpt}...")
        model = model.load_from_checkpoint(ckpt)
        trainer.test(model)

    if hparams.test_mode:
        run_test(model)
    else:
        trainer.fit(model)
        run_test(model)
