import lightning as L
from bonsai.modules.losses.CE import CE
from bonsai.modules.losses.CodeValueLoss import CodeValueLoss
from bonsai.modules.metrics.metrics import SharedPrecisionAtK
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError


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
        if loss_fn == "CE" and self.model.hparams["attn_type"] == "sdpa":
            return CE()
        elif loss_fn == "CE" and self.model.hparams["attn_type"] == "flash":
            from bonsai.modules.losses.CE_FA import CE_FA

            return CE_FA()
        elif loss_fn == "CodeValue" and self.model.hparams["attn_type"] == "sdpa":
            return CodeValueLoss(
                code_loss_fn=CE(),
                **loss_params,
            )
        elif loss_fn == "CodeValue" and self.model.hparams["attn_type"] == "flash":
            from bonsai.modules.losses.CE_FA import CE_FA

            return CodeValueLoss(
                code_loss_fn=CE_FA(),
                **loss_params,
            )

    def training_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.train_loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.val_loss(logits, labels)
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


class ValuePretrainModule(PretrainModule):
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
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            loss_params=loss_params,
            compile_mode=compile_mode,
            learning_rate=learning_rate,
            optimizer_epsilon=optimizer_epsilon,
            scheduler_warmup_epochs=scheduler_warmup_epochs,
        )
        self.value_val_metrics = MetricCollection({"val/MSE": MeanSquaredError()})

    def training_step(self, batch, batch_idx):
        logits, labels, val_logits, val_labels = self.model(batch)
        loss = self.train_loss(logits, val_logits, labels, val_labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, val_logits, val_labels = self.model(batch)
        loss = self.val_loss(logits, val_logits, labels, val_labels)
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics.update(logits, labels)
        self.value_val_metrics.update(val_logits, val_labels)
        self.log_dict(self.val_metrics)
        self.log_dict(self.value_val_metrics)
        return loss
