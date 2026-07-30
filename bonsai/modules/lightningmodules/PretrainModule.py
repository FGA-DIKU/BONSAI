import lightning as L
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection

from bonsai.modules.metrics.metrics import SharedPrecisionAtK

loss_types = {"sdpa": nn.CrossEntropyLoss}
try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss

    loss_types["flash"] = FACrossEntropyLoss
except Exception:
    pass


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

        self.train_loss = loss_types[self.model.hparams["attn_type"]]()
        self.val_loss = nn.CrossEntropyLoss()
        self.val_metrics = self.configure_metrics("val")

        hparams = self.model.hparams.copy()
        hparams.update(
            {
                "learning_rate": learning_rate,
                "optimizer_epsilon": optimizer_epsilon,
                "scheduler_warmup_epochs": scheduler_warmup_epochs,
            }
        )
        hparams.update(self.add_hparams())
        self.save_hyperparameters(hparams)

    def add_hparams(self) -> dict:
        return {}

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

    def compute_loss(self, batch, concept_loss_fn):
        logits, labels = self.model(batch)
        loss = concept_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.compute_loss(batch, self.train_loss)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.compute_loss(batch, self.val_loss)
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
        compile_mode: str = None,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        value_loss_weight: float = 1.0,
    ):
        self.value_loss_weight = value_loss_weight
        super().__init__(
            model=model,
            compile_mode=compile_mode,
            learning_rate=learning_rate,
            optimizer_epsilon=optimizer_epsilon,
            scheduler_warmup_epochs=scheduler_warmup_epochs,
        )
        self.value_loss_fn = nn.MSELoss()

    def add_hparams(self) -> dict:
        return {"value_loss_weight": self.value_loss_weight}

    def compute_loss(self, batch, concept_loss_fn):
        logits, labels, val_logits, val_labels = self.model(batch)
        concept_loss = concept_loss_fn(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        # MLM may mask no numeric tokens in a batch → empty tensors → NaN mean MSE
        if val_logits.numel() == 0:
            value_loss = concept_loss.new_zeros(())
        else:
            value_loss = self.value_loss_fn(val_logits, val_labels)
        loss = concept_loss + self.value_loss_weight * value_loss
        return loss, concept_loss, value_loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, concept_loss, value_loss, _, _ = self.compute_loss(batch, self.train_loss)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/concept_loss", concept_loss, prog_bar=True)
        self.log("train/value_loss", value_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, concept_loss, value_loss, logits, labels = self.compute_loss(
            batch, self.val_loss
        )
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/concept_loss", concept_loss, prog_bar=True)
        self.log("val/value_loss", value_loss, prog_bar=True)
        self.val_metrics.update(logits, labels)
        self.log_dict(self.val_metrics)
        return loss
