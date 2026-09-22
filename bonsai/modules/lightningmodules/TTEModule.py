from os.path import join
from pathlib import Path
from typing import Optional

import polars as pl
import torch
from torch import nn
from torchmetrics import AUROC, AveragePrecision, MetricCollection

from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.metrics.metrics import TimepointMetric


class TTEModule(FinetuneModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        predictions_output_path: Optional[Path] = None,
        prediction_horizons: list = [3, 6, 12, 24, 32],
    ):
        super().__init__(
            model=model,
            compile_mode=compile_mode,
            learning_rate=learning_rate,
            optimizer_epsilon=optimizer_epsilon,
            scheduler_warmup_epochs=scheduler_warmup_epochs,
            predictions_output_path=predictions_output_path,
        )

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        HOURS_PER_MONTH = 365.25 * 24 / 12  # 730.5

        edges = (
            torch.tensor([0] + prediction_horizons, dtype=torch.float32)
            * HOURS_PER_MONTH
        )
        self.lo, self.hi = edges[:-1], edges[1:]  # K = 5 bins

    def configure_metrics(self, prefix: str):
        metrics = {}
        for ihorizon, horizon in enumerate(self.hi):
            metrics[f"{prefix}/AUROC_{horizon}"] = TimepointMetric(
                metric=AUROC(
                    task="binary",
                ),
                timepoint=ihorizon,
            )
            metrics[f"{prefix}/AveragePrecision_{horizon}"] = TimepointMetric(
                metric=AveragePrecision(
                    task="binary",
                ),
                timepoint=ihorizon,
            )
        return MetricCollection(metrics)

    def tte_loss(self, logits, labels, duration):
        at_risk = duration.unsqueeze(-1) > self.lo
        target = (
            (duration.unsqueeze(-1) <= self.hi) & at_risk
        ).float() * labels.unsqueeze(-1)
        loss = (self.bce_loss(logits, target) * at_risk).sum() / at_risk.sum()

        return loss

    def training_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.tte_loss(logits, labels.float(), batch["duration"])
        self.train_metrics(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.tte_loss(logits, labels.float(), batch["duration"])
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics(logits, labels)
        self.log_dict(self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch["target"]
        logits, _ = self.model(batch)
        self.test_metrics(logits, labels)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)

    def on_predict_epoch_start(self) -> None:
        self.predictions = []
        self.labels = []
        self.subject_ids = []
        self.logits = []
        self.duration = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["target"]
        logits = self.model(batch)
        probs = torch.sigmoid(logits)
        if self.predictions_output_path is not None:
            self.logits.append(logits)
            self.labels.append(labels)
            self.subject_ids.append(batch["subject_id"])
            self.predictions.append(probs)
            self.duration.append(batch["duration"])
        return {
            "subject_id": batch["subject_id"],
            "logit": logits,
            "prob": probs,
            "label": labels,
            "duration": batch["duration"],
        }

    def on_predict_epoch_end(self) -> None:
        if self.predictions_output_path is None:
            return

        subject_ids = torch.cat([x.detach().cpu() for x in self.subject_ids])
        logits = torch.cat([x.detach().cpu().float() for x in self.logits])
        labels = torch.cat([x.detach().cpu().long() for x in self.labels])
        probs = torch.cat([x.detach().cpu().float() for x in self.predictions])
        duration = torch.cat(self.duration)

        if self.predictions_output_path is not None:
            self.predictions_output_path.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "subject_id": subject_ids,
                    "prob": torch.cat(
                        [x.detach().cpu().float() for x in self.predictions]
                    ),
                    "duration": duration,
                    **{
                        f"logit_{horizon}": logits[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.hi)
                    },
                    **{
                        f"prob_{horizon}": probs[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.hi)
                    },
                    **{
                        f"label_{horizon}": labels[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.hi)
                    },
                }
            ).write_csv(join(self.predictions_output_path, "predictions.csv"))

            self.predict_metrics.reset()
            metrics = self.predict_metrics(logits, labels)
            metrics = {
                key: float(value.detach().cpu()) for key, value in metrics.items()
            }
            pl.DataFrame(metrics).write_csv(
                join(self.predictions_output_path, "predict_metrics.csv")
            )
