import lightning as L
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchmetrics import MetricCollection
from bonsai.modules.metrics.metrics import SharedPrecisionAtK


class PretrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
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

        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.val_metrics = self.configure_metrics("val")

        hparams = model.config.to_dict()
        hparams.update(
            {
                "learning_rate": learning_rate,
                "optimizer_epsilon": optimizer_epsilon,
                "scheduler_warmup_epochs": scheduler_warmup_epochs,
            }
        )
        self.save_hyperparameters(hparams, ignore=["model"])

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/Precision@1": SharedPrecisionAtK(
                    k=1, max_k=100, reduce="mean"
                ),
                f"{prefix}/Precision@10": SharedPrecisionAtK(
                    k=10, max_k=100, reduce="mean"
                ),
                f"{prefix}/Precision@100": SharedPrecisionAtK(
                    k=100, max_k=100, reduce="mean"
                ),
            },
            compute_groups=[
                [
                    f"{prefix}/Precision@1",
                    f"{prefix}/Precision@10",
                    f"{prefix}/Precision@100",
                ]
            ],
        )

    def training_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.train_loss(
            logits.view(-1, self.model.config.vocab_size), labels.view(-1)
        )
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.val_loss(
            logits.view(-1, self.model.config.vocab_size), labels.view(-1)
        )
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics.update(logits, labels)
        self.log_dict(self.val_metrics)
        return loss

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
