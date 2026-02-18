"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

import lightning as L
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# import logging
# import torch
# from os.path import join
# from corebehrt.functional.io_operations.load import load_vocabulary
# from corebehrt.functional.setup.args import get_args
# from corebehrt.functional.setup.model import load_model_cfg_from_checkpoint
# from corebehrt.functional.trainer.setup import replace_steps_with_epochs
# from corebehrt.main.helper.pretrain import (
#    load_checkpoint_and_epoch,
# )
# from corebehrt.modules.preparation.dataset import MLMDataset, PatientDataset
# from corebehrt.modules.setup.config import load_config
# from corebehrt.modules.setup.directory import DirectoryPreparer
# from corebehrt.modules.setup.initializer import Initializer
# from corebehrt.modules.trainer.trainer import EHRTrainer
# from corebehrt.constants.paths import PREPARED_TRAIN_PATIENTS, PREPARED_VAL_PATIENTS

# CONFIG_PATH = "./corebehrt/configs/pretrain.yaml"


class PretrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        # warmup_epochs: int = None,
        # decoder_warmup_epochs: int = 0,
        # cosine_period_ratio: float = 1,
        # compile_mode: str = None,
        # weights: str = None,
        # load_decoder: bool = True,
        # repeat_stem_weights: bool = True,
        # optimizer: str = "SGD",
        # train_transforms: Optional[transforms.Compose] = None,
        # test_transforms: Optional[transforms.Compose] = None,
        # val_transforms: Optional[transforms.Compose] = None,
        # weight_decay: float = 3e-5,
        # nesterov: bool = True,
        # momentum: float = 0.99,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        """Initialize optimizer from checkpoint or from scratch."""

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
