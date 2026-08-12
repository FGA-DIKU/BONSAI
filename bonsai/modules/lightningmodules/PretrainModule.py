import lightning as L
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection

from bonsai.modules.metrics.metrics import SharedPrecisionAtK
from bonsai.modules.losses.CodeValueLoss import CodeValueLoss
from torchmetrics.regression import MeanSquaredError

loss_types = {"CE": {"sdpa": nn.CrossEntropyLoss}}
try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss

    loss_types["CE"]["flash"] = FACrossEntropyLoss
except Exception:
    pass


class PretrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: str = "CE",
        loss_params: dict = {},
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

        self.train_loss = self.configure_losses(loss_fn, loss_params)
        self.val_loss = self.configure_losses(loss_fn, loss_params)
        self.val_metrics = self.configure_metrics("val")

        hparams = self.model.hparams.copy()
        hparams.update(
            {
                "learning_rate": learning_rate,
                "optimizer_epsilon": optimizer_epsilon,
                "scheduler_warmup_epochs": scheduler_warmup_epochs,
            }
        )
        self.save_hyperparameters(hparams)

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
                f"{prefix}/MSE": MeanSquaredError(),
            },
            compute_groups=[
                [
                    f"{prefix}/Precision@1",
                    f"{prefix}/Precision@10",
                    f"{prefix}/Precision@100",
                ]
            ],
        )

    def configure_losses(self, loss_fn, loss_params):
        if loss_fn == "CE":
            return loss_types["CE"][self.model.hparams["attn_type"]]()
        elif loss_fn == "CodeValue":
            return CodeValueLoss(
                code_loss_fn=loss_types["CE"][self.model.hparams["attn_type"]](),
                **loss_params,
            )

    def training_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.train_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.val_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
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
