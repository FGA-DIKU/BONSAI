from os.path import join
from pathlib import Path
from typing import Optional

import lightning as L
import polars as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import AUROC, Accuracy, AveragePrecision, MetricCollection


class FinetuneModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
        pos_weight: torch.Tensor = None,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        predictions_output_path: Optional[Path] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.predictions_output_path = predictions_output_path

        self.model = model
        if compile_mode is not None:
            self.model.compile(mode=compile_mode)
        self.train_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")
        self.predict_metrics = self.configure_metrics("predict")

        hparams = self.model.hparams.copy()
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
        if self.predictions_output_path is not None:
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
        if self.predictions_output_path is None:
            return

        logits = torch.cat([x.detach().cpu().float() for x in self.logits])
        labels = torch.cat([x.detach().cpu().long() for x in self.labels])

        if self.predictions_output_path is not None:
            self.predictions_output_path.mkdir(parents=True, exist_ok=True)
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
            ).write_csv(join(self.predictions_output_path, "predictions.csv"))

            self.predict_metrics.reset()
            metrics = self.predict_metrics(logits, labels)
            metrics = {
                key: float(value.detach().cpu()) for key, value in metrics.items()
            }
            pl.DataFrame(metrics).write_csv(
                join(self.predictions_output_path, "predict_metrics.csv")
            )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )
        if self.scheduler_warmup_epochs == 0:
            return optimizer

        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=1e-4,
            total_iters=steps_per_epoch * self.scheduler_warmup_epochs,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
