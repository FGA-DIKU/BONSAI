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
        self.prediction_horizons = prediction_horizons
        HOURS_PER_MONTH = 365.25 * 24 / 12  # 730.5
        edges = (
            torch.tensor([0] + prediction_horizons, dtype=torch.float32)
            * HOURS_PER_MONTH
        )
        self.lo, self.hi = edges[:-1], edges[1:]  # K = 5 bins

        super().__init__(
            model=model,
            compile_mode=compile_mode,
            learning_rate=learning_rate,
            optimizer_epsilon=optimizer_epsilon,
            scheduler_warmup_epochs=scheduler_warmup_epochs,
            predictions_output_path=predictions_output_path,
        )

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def configure_metrics(self, prefix: str):
        metrics = {}
        for ihorizon, horizon in enumerate(self.prediction_horizons):
            metrics[f"{prefix}/AUROC_{horizon}mo"] = TimepointMetric(
                metric=AUROC(task="binary", ignore_index=-100),
                timepoint=ihorizon,
            )
            metrics[f"{prefix}/AveragePrecision_{horizon}mo"] = TimepointMetric(
                metric=AveragePrecision(task="binary", ignore_index=-100),
                timepoint=ihorizon,
            )
        return MetricCollection(metrics)

    def tte_targets(self, duration, labels):
        at_risk = duration > self.lo
        target = ((duration <= self.hi) & at_risk).float() * labels
        return target, at_risk

    def tte_loss(self, logits, labels, duration):
        target, at_risk = self.tte_targets(duration, labels)
        loss = (self.bce_loss(logits, target) * at_risk).sum() / at_risk.sum()

        return loss

    def training_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.tte_loss(logits, labels.float(), batch["duration"])
        target, _ = self.tte_targets(batch["duration"], labels.float())
        self.train_metrics(logits, target.int())
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.tte_loss(logits, labels.float(), batch["duration"])
        self.log("val/loss", loss, prog_bar=True)
        target, _ = self.tte_targets(batch["duration"], labels.float())
        self.val_metrics(logits, target.int())
        self.log_dict(self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        target, _ = self.tte_targets(batch["duration"], labels.float())
        self.test_metrics(logits, target.int())
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = {
            "subject_id": [],
            "label": [],
            "duration": [],
            "at_risk": [],
            "logit": [],
            "pred": [],
            "target": [],
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["target"]
        logits = self.model(batch)
        preds = torch.sigmoid(logits)
        targets, at_risk = self.tte_targets(batch["duration"], labels.float())

        if self.predictions_output_path is not None:
            self.predict_outputs["subject_id"].append(batch["subject_id"])
            self.predict_outputs["label"].append(labels)
            self.predict_outputs["at_risk"].append(at_risk)
            self.predict_outputs["duration"].append(batch["duration"])
            self.predict_outputs["logit"].append(logits)
            self.predict_outputs["pred"].append(preds)
            self.predict_outputs["target"].append(targets)
        return {
            "subject_id": batch["subject_id"],
            "label": labels,
            "at_risk": at_risk,
            "duration": batch["duration"],
            "logit": logits,
            "pred": preds,
            "target": targets,
        }

    def on_predict_epoch_end(self) -> None:
        if self.predictions_output_path is None:
            return

        subject_ids = torch.cat(self.predict_outputs["subject_id"])
        labels = torch.cat(self.predict_outputs["label"]).squeeze(-1)
        at_risk = torch.cat(self.predict_outputs["at_risk"]).squeeze(-1)
        duration = torch.cat(self.predict_outputs["duration"]).squeeze(-1)
        logits = torch.cat(self.predict_outputs["logit"]).to(torch.float32)
        preds = torch.cat(self.predict_outputs["pred"]).to(torch.float32)
        targets = torch.cat(self.predict_outputs["target"])

        if self.predictions_output_path is not None:
            self.predictions_output_path.mkdir(parents=True, exist_ok=True)
            df = pl.DataFrame(
                {
                    "subject_id": subject_ids,
                    "label": labels,
                    "duration": duration,
                    **{
                        f"at_risk{horizon}": at_risk[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.prediction_horizons)
                    },
                    **{
                        f"logit_{horizon}": logits[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.prediction_horizons)
                    },
                    **{
                        f"pred_{horizon}": preds[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.prediction_horizons)
                    },
                    **{
                        f"target_{horizon}": targets[:, ihorizon]
                        for ihorizon, horizon in enumerate(self.prediction_horizons)
                    },
                }
            )
            df.write_csv(join(self.predictions_output_path, "predictions.csv"))

            self.predict_metrics.reset()
            metrics = self.predict_metrics(logits, targets.int())
            metrics = {
                key: float(value.detach().cpu()) for key, value in metrics.items()
            }
            pl.DataFrame(metrics).write_csv(
                join(self.predictions_output_path, "predict_metrics.csv")
            )
