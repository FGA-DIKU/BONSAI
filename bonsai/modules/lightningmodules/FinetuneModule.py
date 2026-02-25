import lightning as L
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
        # scheduler_warmup_epochs: int = 0,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.save_hyperparameters(ignore=["model"])
        self.model = (
            torch.compile(model, mode=compile_mode)
            if compile_mode is not None
            else model
        )
        self.train_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")

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
        logits, _ = self.model(batch)
        loss = self.train_loss(logits, labels)
        self.train_metrics(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["target"]
        logits, _ = self.model(batch)
        loss = self.val_loss(logits, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics(logits, labels)
        self.log_dict(self.val_metrics)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
