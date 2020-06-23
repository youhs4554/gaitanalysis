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
from datasets.classification.ucf101 import UCF101
from datasets.classification.hmdb51 import HMDB51
from sklearn.metrics.classification import confusion_matrix

from models import generate_network

from utils.transforms import (
    Compose,
    RandomCrop3D,
    CenterCrop3D,
    RandomHorizontalFlip3D,
    ToTensor3D,
    Normalize3D,
    RandomResizedCrop3D,
    RandomRotation3D,
    Resize3D,
    TemporalRandomCrop,
    TemporalCenterCrop,
    LoopPadding,
    denormalize,
)
from utils.callbacks import *

import sklearn.metrics
import os
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

warnings.filterwarnings("ignore")


def collate_fn_multipositon_prediction(dataset_iter):
    """
        Collation function for multi-positional (temporal) cropping,
        to guirantee max_clip_lenght among a batch (same-pad?), fill shorter frames with repeated frames
    """

    nclips_max = max([len(sample[0]) for sample in dataset_iter])

    batch = []
    for sample in dataset_iter:
        video_stack, mask_stack, label = sample
        nclips = video_stack.size(0)

        # video_stack : (nclips,C,D,H,W)
        repeated_pts = LoopPadding(nclips_max)(torch.arange(0, nclips).tolist())[
            : nclips_max - nclips
        ]

        # merge with repeated frames
        video_stack = torch.cat((video_stack, video_stack[repeated_pts]), axis=0)
        mask_stack = torch.cat((mask_stack, mask_stack[repeated_pts]), axis=0)

        batch.append((video_stack, mask_stack, label))

    batch_transposed = list(zip(*batch))
    for i in range(len(batch_transposed)):
        if isinstance(batch_transposed[i][0], torch.Tensor):
            batch_transposed[i] = torch.stack(batch_transposed[i], 0)
    for i in range(2):
        bs, nclips, *cdhw = batch_transposed[i].shape
        batch_transposed[i] = batch_transposed[i].view(
            -1, *cdhw
        )  # (batch*nclips, c,d,h,w)

    return tuple(batch_transposed)


def mycollaten_fn(dataset_iter):
    max_boxes = 0
    for sample in dataset_iter:
        video, mask, coord, label = sample
        local_max = reduce(lambda a, b: b if b > a else a, [len(x) for x in coord])
        if local_max > max_boxes:
            max_boxes = local_max

    batch = []
    for sample in dataset_iter:
        video, mask, coord, label = sample

        for i in range(len(coord)):
            padded_coord = torch.nn.functional.pad(
                torch.tensor(coord[i]).float(), (0, 0, 0, max_boxes - len(coord[i]))
            )
            coord[i] = padded_coord / mask.size(2)  # normalization with mask size
        # stack coord
        coord = torch.stack(coord)
        batch.append((video, mask, coord, label))

    batch_transposed = list(zip(*batch))
    for i in range(len(batch_transposed)):
        if isinstance(batch_transposed[i][0], torch.Tensor):
            batch_transposed[i] = torch.stack(batch_transposed[i], 0)

    coord_series = []
    batch_size = len(dataset_iter)
    for t in range(len(batch_transposed[2][0])):
        box_coord = []
        for b in range(batch_size):
            box_coord.append(batch_transposed[2][b][t])
        coord_series.append(torch.stack(box_coord))
    batch_transposed[2] = torch.stack(coord_series)

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

    def __init__(self, hparams, test_mode=False):
        super(LightningVideoClassifier, self).__init__()

        self.hparams = hparams
        self.test_mode = test_mode

        name = hparams.dataset  # name of dataset

        if name not in ["UCF101", "HMDB51"]:
            raise ValueError(
                "Unsupported Dataset. This class only supports ( UCF101 | HMDB51 )"
            )

        n_outputs = int("".join([c for c in name if c.isdigit()]))

        self.dataset_init_func = self.datasets_map.get(name)
        self.model = generate_network(hparams, n_outputs=n_outputs)

    def forward(self, *batch, averaged=None):
        video, mask, coord, label = batch

        if self.trainer.global_step % 50 == 0:
            v = video[0].permute(1, 2, 3, 0)
            v = denormalize(v, self.hparams.mean, self.hparams.std)

            m = mask[0].permute(1, 2, 3, 0).repeat(1, 1, 1, 3)

            i = torch.cat((v, m), 0)
            self.logger.experiment.add_images(
                "clip_batch_image_and_mask",
                i,
                self.trainer.global_step,
                dataformats="NHWC",
            )

        out, loss_dict = self.model(
            video, mask, coord, targets=label, averaged=averaged
        )

        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}

        return out, loss_dict

    def step(self, batch, batch_idx, averaged=None):
        video, mask, coord, label = batch

        out, loss_dict = self.forward(video, mask, coord, label, averaged=averaged)

        loss = sum(loss for loss in loss_dict.values())
        acc = (out.argmax(1) == label).float().mean()

        return {
            "loss": loss,
            "acc": acc,
            "loss_dict": loss_dict,
            "pred": out.argmax(1).float(),
            "label": label.float(),
        }

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx)

    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, batch_idx, averaged=True)  # for averaged logits

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
        return {"test_loss": avg_loss, "test_acc": avg_acc, "test_cm": cm}

    def prepare_data(self):
        center_crop = Compose(
            [
                CenterCrop3D((self.hparams.sample_size, self.hparams.sample_size)),
                ToTensor3D(),
            ]
        )

        spatial_transform = {
            "train": Compose(
                [
                    RandomHorizontalFlip3D(),
                    # RandomRotation3D(
                    #     transform2D=transforms.RandomRotation(45)),
                    RandomCrop3D(
                        transform2D=transforms.RandomCrop(
                            size=(self.hparams.sample_size, self.hparams.sample_size)
                        )
                    ),
                    ToTensor3D(),
                ]
            ),
            "val": Compose(
                [
                    CenterCrop3D(
                        size=(self.hparams.sample_size, self.hparams.sample_size)
                    ),
                    ToTensor3D(),
                ]
            ),
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
            "test": None,
        }

        norm_method = Normalize3D(mean=self.hparams.mean, std=self.hparams.std)

        self.train_ds = self.dataset_init_func(
            root=self.hparams.data_root,
            annotation_path=self.hparams.annotation_file,
            detection_file_path=self.hparams.detection_file,
            sample_rate=1 if self.hparams.detection_file.endswith("_full") else 5,
            input_size=(self.hparams.sample_duration, 128, 171),
            train=True,
            fold=1,
            temporal_transform=temporal_transform["train"],
            spatial_transform=spatial_transform["train"],
            num_workers=self.hparams.n_threads,
            norm_method=norm_method,
            sample_unit="video",
        )

        self.test_ds = self.dataset_init_func(
            root=self.hparams.data_root,
            annotation_path=self.hparams.annotation_file,
            detection_file_path=self.hparams.detection_file,
            sample_rate=1 if self.hparams.detection_file.endswith("_full") else 5,
            input_size=(self.hparams.sample_duration, 128, 171),
            train=False,
            fold=1,
            temporal_transform=None,
            spatial_transform=spatial_transform["test" if self.test_mode else "val"],
            num_workers=self.hparams.n_threads,
            norm_method=norm_method,
            sample_unit="video" if self.test_mode else "clip",
        )

        print("train : {}, test : {}".format(len(self.train_ds), len(self.test_ds)))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.n_threads,
            collate_fn=mycollaten_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.n_threads,
            collate_fn=mycollaten_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=30,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.n_threads,
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
    ):
        # use lr proposed by lr_finder
        lr = self.hparams.learning_rate or self.lr
        until = 5 * len(self.train_dataloader())  # warm-up for 5 epochs

        # warm up lr
        if self.trainer.global_step < until:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / until)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizers = [
            torch.optim.SGD(
                self.parameters(),
                lr=(self.hparams.learning_rate or self.lr),
                momentum=0.9,
                nesterov=True,
                weight_decay=self.hparams.weight_decay,
            ),
            # Ranger(
            #     self.parameters(),
            #     lr=(self.hparams.learning_rate or self.lr),
            #     weight_decay=self.hparams.weight_decay,
            # )
        ]
        schedulers = [
            {
                # "scheduler": torch.optim.lr_scheduler.StepLR(
                #     optimizers[0], 10, gamma=0.5
                # ),
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], T_max=50
                ),
                "interval": "step",
                "frequency": len(self.train_dataloader()),
            },
            # {
            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizers[0], "min", factor=0.5, patience=5, verbose=True
            #     ),
            #     "monitor": "val_loss",  # Default: val_loss
            #     "interval": "step",
            #     "frequency": len(self.train_dataloader()) + len(self.val_dataloader()),
            # }
        ]

        return optimizers, schedulers


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    # for reproductivity
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--ckpt", type=str, help="path to trained(or pre-trained) ckpt")
    args = parser.parse_args()

    if not os.path.exists(args.cfg_file):
        raise ValueError("Configuration file {} is not exist...".format(args.cfg_file))

    with open(args.cfg_file) as file:
        # load hparams
        hparams = json.load(file)
        hparams = argparse.Namespace(**hparams)

    model = LightningVideoClassifier(hparams, test_mode=args.test_mode)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=20, verbose=False, mode="max"
    )

    lr_logger = LearningRateLogger()

    # init logger
    base_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs", name=f"{hparams.model_arch}_{hparams.backbone}"
    )
    tb_logger = TensorBoard_Logger(base_logger)

    trainer = pl.Trainer(
        gpus=5,
        distributed_backend="dp",
        callbacks=[lr_logger, tb_logger],
        early_stop_callback=early_stop_callback,
        logger=base_logger,
        log_save_interval=10,
        max_epochs=100,
        # auto_lr_find=True,
        # auto_scale_batch_size="power",  #  train : 100, test : x is optimal,
    )

    if args.test_mode:
        model = model.load_from_checkpoint(args.ckpt, test_mode=args.test_mode)
        trainer.test(model)
    else:
        trainer.fit(model)
