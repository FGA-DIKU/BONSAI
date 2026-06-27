from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from lightning.pytorch.callbacks import Callback


class LossPlotCallback(Callback):
    def __init__(self, metrics_csv: Path, save_path: Path):
        self.metrics_csv = metrics_csv
        self.save_path = save_path

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.metrics_csv.exists():
            return

        df = pl.read_csv(self.metrics_csv)
        if df.is_empty():
            return

        train = (
            df.filter(pl.col("train/loss").is_not_null())
            .group_by("epoch")
            .agg(pl.col("train/loss").last())
            .sort("epoch")
        )
        val = (
            df.filter(pl.col("val/loss").is_not_null())
            .group_by("epoch")
            .agg(pl.col("val/loss").last())
            .sort("epoch")
        )
        if train.is_empty() and val.is_empty():
            return

        fig, ax = plt.subplots()
        if not train.is_empty():
            ax.plot(
                train["epoch"].to_list(),
                train["train/loss"].to_list(),
                label="train",
            )
        if not val.is_empty():
            ax.plot(
                val["epoch"].to_list(),
                val["val/loss"].to_list(),
                label="val",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training and validation loss")
        ax.legend()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
