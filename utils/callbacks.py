import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning import Callback


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

    def on_batch_end(self, trainer, pl_module):
        self.write_logs(trainer, pl_module, name="train")

    def on_validation_end(self, trainer, pl_module):
        self.write_logs(trainer, pl_module, name="valid")

    def on_test_end(self, trainer, pl_module):
        # For test mode, we jues draw confusion matrix with accuracy is depicted
        acc = trainer.callback_metrics.get("test_acc")
        cm = trainer.callback_metrics.get("test_cm")

        # Normalise
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        nclasses = len(cm)
        df_cm = pd.DataFrame(cm, range(nclasses), range(nclasses))

        fig = plt.figure(figsize=(10, 7), dpi=300)
        ax = fig.add_subplot(111)

        sns.set(font_scale=0.5)  # for label size
        sns.heatmap(
            df_cm, ax=ax, xticklabels=range(nclasses), yticklabels=range(nclasses)
        )  # font size
        ax.set_title(f"Confusion Matrix (ACC={acc.item():.4f})")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

        self.base_logger.experiment.add_figure(
            "confusion_matrix (test)", fig, global_step=0  # dummy global-step
        )
