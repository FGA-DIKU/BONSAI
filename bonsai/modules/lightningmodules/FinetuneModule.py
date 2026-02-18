import lightning as L
import torch
from abc import abstractmethod
from torch import nn
# from torch.optim import AdamW
# from transformers import get_linear_schedule_with_warmup


class FinetuneModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
        pos_weight: 
        # learning_rate: float = 5e-4,
        # loss_fn: nn.Module = nn.CrossEntropyLoss(),
        # optimizer_epsilon: float = 1e-6,
        # scheduler_warmup_epochs: int = 0,
    ):
        super().__init__()
        # self.learning_rate = learning_rate
        # self.optimizer_epsilon = optimizer_epsilon
        self.save_hyperparameters(ignore=["model"])
        self.model = (
            torch.compile(model, mode=compile_mode)
            if compile_mode is not None
            else model
        )
        self.train_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        # optimizer = AdamW(
        #    self.model.parameters(),
        #    lr=self.learning_rate,
        #    eps=self.optimizer_epsilon,
        # )
        # steps_per_epoch = (
        #    self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # )
        # scheduler = get_linear_schedule_with_warmup(
        #    optimizer=optimizer,
        #    num_warmup_steps=steps_per_epoch * self.scheduler_warmup_epochs,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        # scheduler_config = {
        #    "scheduler": scheduler,
        #    "interval": "step",
        #    "frequency": 1,
        # }
        return [optimizer], [scheduler_config]
