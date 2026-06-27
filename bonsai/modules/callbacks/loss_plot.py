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
        self.sens_save_path = save_path.with_name("sens_at_spec85.png")

    @staticmethod
    def _epoch_series(df: pl.DataFrame, column: str) -> pl.DataFrame:
        if column not in df.columns:
            return pl.DataFrame()
        return (
            df.filter(pl.col(column).is_not_null())
            .group_by("epoch")
            .agg(pl.col(column).last())
            .sort("epoch")
        )

    @staticmethod
    def _plot_epoch_series(
        train: pl.DataFrame,
        val: pl.DataFrame,
        *,
        train_col: str,
        val_col: str,
        ylabel: str,
        title: str,
        save_path: Path,
    ) -> None:
        if train.is_empty() and val.is_empty():
            return

        fig, ax = plt.subplots()
        if not train.is_empty():
            ax.plot(
                train["epoch"].to_list(),
                train[train_col].to_list(),
                label="train",
            )
        if not val.is_empty():
            ax.plot(
                val["epoch"].to_list(),
                val[val_col].to_list(),
                label="val",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.metrics_csv.exists():
            return

        df = pl.read_csv(self.metrics_csv)
        if df.is_empty():
            return

        self._plot_epoch_series(
            self._epoch_series(df, "train/loss"),
            self._epoch_series(df, "val/loss"),
            train_col="train/loss",
            val_col="val/loss",
            ylabel="loss",
            title="Training and validation loss",
            save_path=self.save_path,
        )
        self._plot_epoch_series(
            self._epoch_series(df, "train/sens@spec85"),
            self._epoch_series(df, "val/sens@spec85"),
            train_col="train/sens@spec85",
            val_col="val/sens@spec85",
            ylabel="sensitivity @ specificity 0.85",
            title="Training and validation sensitivity @ specificity 0.85",
            save_path=self.sens_save_path,
        )
