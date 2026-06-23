from pathlib import Path
from typing import Optional

import lightning as L
import polars as pl
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchmetrics import MetricCollection, Accuracy, AUROC, AveragePrecision


class FinetuneModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
        pos_weight: torch.Tensor = None,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        predictions_path: Optional[Path] = None,
        test_metrics_path: Optional[Path] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.predictions_path = predictions_path
        self.test_metrics_path = test_metrics_path

        self.model = model
        if compile_mode is not None:
            self.model.compile(mode=compile_mode)
        self.train_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")
        hparams = model.config.to_dict()
        hparams.update(
            {
                "learning_rate": learning_rate,
                "optimizer_epsilon": optimizer_epsilon,
                "scheduler_warmup_epochs": scheduler_warmup_epochs,
                "pos_weight": None if pos_weight is None else pos_weight.item(),
            }
        )
        self.save_hyperparameters(hparams)

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/Accuracy": Accuracy(
                    task="binary",
                    threshold=0.6,
                ),
                f"{prefix}/AUROC": AUROC(
                    task="binary",
                ),
                f"{prefix}/AveragePrecision": AveragePrecision(
                    task="binary",
                ),
            },
        )

    def training_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.train_loss(logits, labels.float())
        self.train_metrics(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.val_loss(logits, labels.float())
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["target"]
        logits = self.model(batch)
        probs = torch.sigmoid(logits)
        if self.predictions_path is not None or self.test_metrics_path is not None:
            self.logits.append(logits.squeeze(-1))
            self.labels.append(labels.squeeze(-1))
            self.subject_ids.append(batch["subject_id"])
            self.predictions.append(probs.squeeze(-1))
        return {
            "subject_id": batch["subject_id"],
            "logit": logits.squeeze(-1),
            "prob": probs.squeeze(-1),
            "label": labels.squeeze(-1),
        }

    def on_predict_epoch_end(self) -> None:
        if self.predictions_path is None and self.test_metrics_path is None:
            return

        logits = torch.cat([x.detach().cpu().float() for x in self.logits])
        labels = torch.cat([x.detach().cpu().long() for x in self.labels])

        if self.predictions_path is not None:
            self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "subject_id": torch.cat(
                        [x.detach().cpu() for x in self.subject_ids]
                    ),
                    "logit": logits,
                    "prob": torch.cat(
                        [x.detach().cpu().float() for x in self.predictions]
                    ),
                    "label": labels,
                }
            ).write_csv(self.predictions_path)

        if self.test_metrics_path is not None:
            self.test_metrics.reset()
            metrics = self.test_metrics(logits, labels)
            metrics = {
                key: float(value.detach().cpu())
                for key, value in metrics.items()
            }
            self.test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(metrics).write_csv(self.test_metrics_path)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )
        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * self.scheduler_warmup_epochs,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
