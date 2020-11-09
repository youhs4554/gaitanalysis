import torch
import pytorch_lightning as pl
from models import generate_network
import numpy as np
from collections import defaultdict

from utils.transforms import denormalize


def mixup_data(video, mask, label, clip_length, alpha=0.5):
    lambda_ = np.random.beta(alpha, alpha)

    batch_size = video.size(0)
    indices = torch.randperm(batch_size)

    mixed_video = lambda_ * video + (1 - lambda_) * video[indices]
    mixed_mask = lambda_ * mask + (1 - lambda_) * mask[indices]

    label_a, label_b = label, label[indices]
    return mixed_video, mixed_mask, label_a, label_b, lambda_


class LightningVideoClassifier(pl.LightningModule):
    # dataset statistics
    MEAN = [0.43216, 0.394666, 0.37645]
    STD = [0.22803, 0.22145, 0.216989]

    def __init__(self, hparams):
        super(LightningVideoClassifier, self).__init__()

        self.hparams = hparams
        name = hparams.dataset  # name of dataset

        if name not in ["UCF101", "HMDB51", "CesleaFDD6"]:
            raise ValueError(
                "Unsupported Dataset. This class only supports ( UCF101 | HMDB51 | CesleaFDD6 )"
            )
        if hparams.task == "classification":
            n_outputs = int("".join([c for c in name if c.isdigit()]))
        elif hparams.task == "regression":
            # if hparams.dataset == "GAIT":
            from cfg.target_columns import BASIC_GAIT_PARAMS, ADVANCED_GAIT_PARAMS

            n_outputs = len(BASIC_GAIT_PARAMS)
            if hparams.model_arch == "ConcatenatedSTCNet":
                n_outputs = len(ADVANCED_GAIT_PARAMS)

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
            v = denormalize(v, self.MEAN, self.STD)

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

        avg_top5_acc = torch.stack([x["top5_acc"].mean() for x in outputs]).mean()

        return {
            "test_loss": avg_loss,
            "test_acc": avg_acc,
            "test_top5_acc": avg_top5_acc,
        }

    def configure_optimizers(self):
        base_lr = self.hparams.learning_rate or self.lr

        optimizers = [
            torch.optim.SGD(
                self.parameters(),
                lr=base_lr,
                momentum=0.95,
                weight_decay=self.hparams.weight_decay,
            )
        ]
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0], "max", patience=5, verbose=True
                ),
                "interval": "epoch",
                "monitor": "val_acc",
            }
        ]

        return optimizers, schedulers


if __name__ == "__main__":
    pass
