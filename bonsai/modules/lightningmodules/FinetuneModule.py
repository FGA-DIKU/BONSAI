import lightning as L
import torch
from abc import abstractmethod
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


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

    """ add metrics:
    metrics:
        accuracy:
            _target_: corebehrt.modules.monitoring.metrics.Accuracy
            threshold: 0.6
        roc_auc:
            _target_: corebehrt.modules.monitoring.metrics.ROC_AUC
        pr_auc:
            _target_: corebehrt.modules.monitoring.metrics.PR_AUC
        precentage_positives:
            _target_: corebehrt.modules.monitoring.metrics.Percentage_Positives

        mean_probability:
            _target_: corebehrt.modules.monitoring.metrics.Mean_Probability
        true_positives:
            _target_: corebehrt.modules.monitoring.metrics.True_Positives

        true_negatives:
            _target_: corebehrt.modules.monitoring.metrics.True_Negatives
        false_positives:
            _target_: corebehrt.modules.monitoring.metrics.False_Positives

        false_negatives:
            _target_: corebehrt.modules.monitoring.metrics.False_Negatives
    """

    @abstractmethod
    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.train_loss(logits, batch["labels"])
        return loss

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

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
