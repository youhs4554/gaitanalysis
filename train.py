import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import argparse
from pytorch_lightning.callbacks import LearningRateLogger
from datasets.classification.ucf101 import UCF101
from datasets.classification.hmdb51 import HMDB51

from models import generate_network

from utils.transforms import (
    Compose, RandomCrop3D, CenterCrop3D, RandomHorizontalFlip3D, ToTensor3D, Normalize3D,
    TemporalRandomCrop, TemporalCenterCrop, LoopPadding)

import sklearn.metrics
import os
import json
import argparse
import numpy as np
import random
import warnings
from ranger import Ranger  # this is from ranger.py
from collections import namedtuple

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

        # for reproductivity
        set_seed(0)

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
        out, loss_dict = self.model(
            video, mask, targets=label, averaged=averaged)

        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}

        return out, loss_dict

    def training_step(self, train_batch, batch_idx):
        video, mask, label = train_batch
        out, loss_dict = self.forward(video, mask, label)

        loss = sum(loss for loss in loss_dict.values())
        acc = (out.argmax(1) == label).float().mean()

        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        video_stack, mask_stack, label = val_batch
        bs, nclips, *rgb_cdhw = video_stack.size()
        bs, nclips, *mask_cdhw = mask_stack.size()

        out, loss_dict = self.forward(
            video_stack.view(-1, *rgb_cdhw), mask_stack.view(-1, *mask_cdhw), label, averaged=True)
        loss = sum(loss for loss in loss_dict.values())
        acc = (out.argmax(1) == label).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].mean() for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        center_crop = Compose([
            CenterCrop3D((self.hparams.sample_size,
                          self.hparams.sample_size)),
            ToTensor3D()
        ])

        spatial_transform = {
            "train": Compose(
                [
                    RandomCrop3D(transform2D=transforms.RandomCrop(
                        size=(self.hparams.sample_size, self.hparams.sample_size))
                    ),
                    RandomHorizontalFlip3D(),
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
        params_with_no_wd = []
        for m in self.modules():
            if m.__class__.__name__.find('BatchNorm') != -1:
                params_with_no_wd += list(m.parameters())

        optimizers = [
            torch.optim.SGD(self.parameters(),
                            lr=(self.hparams.learning_rate or self.lr),
                            momentum=0.9, nesterov=True,
                            weight_decay=self.hparams.weight_decay
                            )
        ]
        schedulers = [
            torch.optim.lr_scheduler.StepLR(
                optimizers[0], 7, gamma=0.1)
            # {
            #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizers[0], 'max',
            #         factor=0.1,
            #         patience=5, verbose=True),
            #     'monitor': 'val_acc',  # Default: val_loss
            #     'interval': 'step',
            #     'frequency': len(self.train_dataloader()) + len(self.val_dataloader())
            # },
        ]

        return optimizers, schedulers


if __name__ == "__main__":

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
    trainer = pl.Trainer(gpus=5, distributed_backend='dp', callbacks=[lr_logger],
                         early_stop_callback=early_stop_callback,
                         #  auto_lr_find=True,
                         log_save_interval=10
                         )

    trainer.fit(model)
