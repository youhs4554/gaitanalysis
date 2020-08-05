import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
import torch


class TensorBoard_Logger(Callback):
    def __init__(self, base_logger):
        super().__init__()
        self.base_logger = base_logger

    def _write_logs(self, prefix_tag, name, tag_scalar_dict, global_step):
        for k in tag_scalar_dict:
            val = tag_scalar_dict[k].item()
            if len(name) > 0:
                tagname = f"{prefix_tag}/{name}/{k}"
            else:
                tagname = f"{prefix_tag}"
            self.base_logger.experiment.add_scalar(
                tagname, tag_scalar_dict[k], global_step
            )

    def write_logs(self, trainer, pl_module, name):
        logs = trainer.callback_metrics
        if not logs:
            return
        loss_dict = logs["loss_dict"]

        log_save_interval = trainer.log_save_interval if name == "train" else 1

        if trainer.global_step % log_save_interval == 0:
            self._write_logs("Losses", name, loss_dict, trainer.global_step)
            self._write_logs(
                "Accuracy",
                name,
                {"acc": logs["acc" if name == "train" else "val_acc"]},
                trainer.global_step,
            )
            if name == "train":
                tagname = f"lr-{pl_module.optimizer.__class__.__name__}"
                self._write_logs(
                    tagname,
                    "",
                    {tagname: torch.tensor(pl_module.optimizer.param_groups[0]["lr"])},
                    trainer.global_step,
                )

    def on_batch_end(self, trainer, pl_module):
        self.write_logs(trainer, pl_module, name="train")

    def on_validation_end(self, trainer, pl_module):
        self.write_logs(trainer, pl_module, name="valid")

    def on_test_end(self, trainer, pl_module):
        pass