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

from models import generate_network

from utils.transforms import (
    Compose, RandomCrop3D, CenterCrop3D, RandomHorizontalFlip3D, ToTensor3D, Normalize3D, RandomResizedCrop3D, RandomRotation3D, Resize3D,
    TemporalRandomCrop, TemporalCenterCrop, LoopPadding, denormalize)

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
        repeated_pts = LoopPadding(nclips_max)(
            torch.arange(0, nclips).tolist())[:nclips_max-nclips]

        # merge with repeated frames
        video_stack = torch.cat(
            (video_stack,  video_stack[repeated_pts]), axis=0)
        mask_stack = torch.cat(
            (mask_stack, mask_stack[repeated_pts]), axis=0)

        batch.append(
            (video_stack, mask_stack, label)
        )

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
    datasets_map = {
        'UCF101': UCF101,
        'HMDB51': HMDB51
    }

    def __init__(self, hparams):
        super(LightningVideoClassifier, self).__init__()

        self.hparams = hparams

        name = hparams.dataset  # name of dataset

        if name not in ['UCF101', 'HMDB51']:
            raise ValueError(
                "Unsupported Dataset. This class only supports ( UCF101 | HMDB51 )")

        n_outputs = int(''.join([c for c in name if c.isdigit()]))

        self.dataset_init_func = self.datasets_map.get(name)
        self.model = generate_network(
            hparams, n_outputs=n_outputs)

    def forward(self, *batch, averaged=None):
        video, mask, label = batch

        if self.trainer.global_step % 100 == 0:
            v = video[0].permute(
                1, 2, 3, 0)
            v = denormalize(v, self.hparams.mean, self.hparams.std)

            m = mask[0].permute(
                1, 2, 3, 0).repeat(1, 1, 1, 3)

            i = torch.cat((v, m), 0)
            self.logger.experiment.add_images(
                'clip_batch_image_and_mask', i, self.trainer.global_step, dataformats='NHWC')

        out, loss_dict = self.model(
            video, mask, targets=label, averaged=averaged)

        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}

        return out, loss_dict

    def training_step(self, train_batch, batch_idx):
        video, mask, label = train_batch
        out, loss_dict = self.forward(video, mask, label)

        loss = sum(loss for loss in loss_dict.values())
        acc = (out.argmax(1) == label).float().mean()

        return {'loss': loss, 'acc': acc, 'loss_dict': loss_dict}

    def validation_step(self, val_batch, batch_idx):
        video_stack, mask_stack, label = val_batch
        bs, nclips, *rgb_cdhw = video_stack.size()
        bs, nclips, *mask_cdhw = mask_stack.size()

        if torch.no_grad():
            out, loss_dict = self.forward(
                video_stack.view(-1, *rgb_cdhw), mask_stack.view(-1, *mask_cdhw), label, averaged=True)

        loss = sum(loss for loss in loss_dict.values())
        acc = (out.argmax(1) == label).float().mean()

        return {'val_loss': loss, 'val_acc': acc, 'loss_dict': loss_dict}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].mean() for x in outputs]).mean()

        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def prepare_data(self):
        center_crop = Compose([
            CenterCrop3D((self.hparams.sample_size,
                          self.hparams.sample_size)),
            ToTensor3D()
        ])

        spatial_transform = {
            "train": Compose(
                [
                    RandomHorizontalFlip3D(),
                    RandomRotation3D(
                        transform2D=transforms.RandomRotation(45)),
                    RandomCrop3D(transform2D=transforms.RandomCrop(
                        size=(self.hparams.sample_size, self.hparams.sample_size))
                    ),
                    ToTensor3D()
                ]
            ),
            "test": Compose(
                [
                    transforms.Lambda(lambda clips: torch.stack(
                        [center_crop(clip) for clip in clips]))
                ]

            )
        }

        temporal_transform = {
            "train": TemporalRandomCrop(size=self.hparams.sample_duration),
            "test": None
        }

        norm_method = Normalize3D(mean=self.hparams.mean, std=self.hparams.std)

        self.train_ds = self.dataset_init_func(root=self.hparams.data_root,
                                               annotation_path=self.hparams.annotation_file,
                                               detection_file_path=self.hparams.detection_file,
                                               sample_rate=5, input_size=(self.hparams.sample_duration, 128, 171), train=True, fold=1,
                                               temporal_transform=temporal_transform["train"],
                                               spatial_transform=spatial_transform["train"],
                                               norm_method=norm_method)

        self.val_ds = self.dataset_init_func(root=self.hparams.data_root,
                                             annotation_path=self.hparams.annotation_file,
                                             detection_file_path=self.hparams.detection_file,
                                             sample_rate=5, input_size=(self.hparams.sample_duration, 128, 171), train=False, fold=1,
                                             temporal_transform=temporal_transform["test"],
                                             spatial_transform=spatial_transform["test"],
                                             norm_method=norm_method)
        print("trian : {}, valid : {}".format(
            len(self.train_ds), len(self.val_ds)))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True, num_workers=self.hparams.n_threads)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True, num_workers=self.hparams.n_threads, collate_fn=collate_fn_multipositon_prediction)

#   def test_dataloader(self):
#     return DataLoader(self,mnist_test, batch_size=64)

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_idx, optimizer,
                       optimizer_idx, second_order_closure=None):
        # use lr proposed by lr_finder
        lr = (self.hparams.learning_rate or self.lr)
        until = 5 * len(self.train_dataloader())  # warm-up for 5 epochs

        # warm up lr
        if self.trainer.global_step < until:
            lr_scale = min(1.0, float(self.trainer.global_step+1) / until)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizers = [
            torch.optim.SGD(self.parameters(),
                            lr=(self.hparams.learning_rate or self.lr),
                            momentum=0.9, nesterov=True,
                            weight_decay=self.hparams.weight_decay
                            ),

        ]
        schedulers = [
            {
                # "scheduler": torch.optim.lr_scheduler.StepLR(
                #     optimizers[0], 7, gamma=0.1),
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=50),
                "interval": "step",
                "frequency": len(self.train_dataloader())
            },
            # {
            #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizers[0], 'max',
            #         factor=0.1,
            #         patience=10, verbose=True),
            #     'monitor': 'val_acc',  # Default: val_loss
            #     'interval': 'step',
            #     'frequency': len(self.train_dataloader()) + len(self.val_dataloader())
            # },
        ]

        return optimizers, schedulers


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    # for reproductivity
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file')
    args = parser.parse_args()
    if not os.path.exists(args.cfg_file):
        raise ValueError(
            "Configuration file {} is not exist...".format(args.cfg_file))

    with open(args.cfg_file) as file:
        # load hparams
        hparams = json.load(file)
        hparams = argparse.Namespace(**hparams)

    # train
    model = LightningVideoClassifier(hparams)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='max'
    )

    lr_logger = LearningRateLogger()

    class TensorBoard_Logger(Callback):
        def __init__(self, base_logger):
            super().__init__()
            self.base_logger = base_logger

        def _write_logs(self, prefix_tag, name, tag_scalar_dict, global_step):
            for k in tag_scalar_dict:
                val = tag_scalar_dict[k].item()
                self.base_logger.experiment.add_scalar(
                    f"{prefix_tag}/{name}/{k}", tag_scalar_dict[k], global_step
                )

        def write_logs(self, trainer, pl_module, name):
            logs = trainer.callback_metrics
            if not logs:
                return
            loss_dict = logs['loss_dict']

            log_save_interval = trainer.log_save_interval if name == 'train' else 1

            if trainer.global_step % log_save_interval == 0:
                self._write_logs('Losses', name, loss_dict,
                                 trainer.global_step)
                self._write_logs(
                    'Accuracy', name, {'acc': logs['acc' if name == 'train' else 'val_acc']}, trainer.global_step)

        def on_batch_end(self, trainer, pl_module):
            self.write_logs(trainer, pl_module, name='train')

        def on_validation_end(self, trainer, pl_module):
            self.write_logs(trainer, pl_module, name='valid')

    # init logger
    base_logger = pl.loggers.TensorBoardLogger(
        'lightning_logs', name=f'{hparams.model_arch}_{hparams.backbone})')
    tb_logger = TensorBoard_Logger(base_logger)

    trainer = pl.Trainer(gpus=5, distributed_backend='dp', callbacks=[lr_logger, tb_logger],
                         early_stop_callback=early_stop_callback,
                         logger=base_logger,
                         log_save_interval=10,
                         )
    trainer.fit(model)
