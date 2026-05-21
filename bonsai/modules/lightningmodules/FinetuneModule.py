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
        scheduler_warmup_epochs: int = 0,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs

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
