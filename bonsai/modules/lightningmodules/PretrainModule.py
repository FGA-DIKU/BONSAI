"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

import lightning as L
import logging
import torch
from os.path import join
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.setup.model import load_model_cfg_from_checkpoint
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.main.helper.pretrain import (
    load_checkpoint_and_epoch,
)
from corebehrt.modules.preparation.dataset import MLMDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.initializer import Initializer
from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.constants.paths import PREPARED_TRAIN_PATIENTS, PREPARED_VAL_PATIENTS

CONFIG_PATH = "./corebehrt/configs/pretrain.yaml"


class PretrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        # learning_rate: float = 1e-3,
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


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_pretrain()

    logger = logging.getLogger("pretrain")

    # Check if we are training from checkpoint, if so, update model config
    restart_path = cfg.paths.get("restart_model")
    if restart_path:
        cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # Get data
    train_data = PatientDataset(
        torch.load(join(cfg.paths.prepared_data, PREPARED_TRAIN_PATIENTS))
    )
    val_data = PatientDataset(
        torch.load(join(cfg.paths.prepared_data, PREPARED_VAL_PATIENTS))
    )
    vocab = load_vocabulary(cfg.paths.prepared_data)

    # Initialize datasets
    train_dataset = MLMDataset(train_data.patients, vocab, **cfg.data.dataset)
    val_dataset = MLMDataset(val_data.patients, vocab, **cfg.data.dataset)

    if "scheduler" in cfg:
        logger.info("Replacing steps with epochs in scheduler config")
        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.batch_size, len(train_dataset)
        )

    checkpoint, epoch = None, None
    if restart_path:
        checkpoint, epoch = load_checkpoint_and_epoch(
            restart_path, cfg.paths.get("checkpoint_epoch")
        )
        logger.info(f"Continue training from epoch {epoch}")
    initializer = Initializer(cfg, checkpoint=checkpoint, model_path=restart_path)
    model = initializer.initialize_pretrain_model(train_dataset)
    logger.info("Initializing optimizer")
    optimizer = initializer.initialize_optimizer(model)
    scheduler = initializer.initialize_scheduler(optimizer)

    logger.info("Initialize trainer")
    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        last_epoch=epoch,
    )
    logger.info("Start training")
    trainer.train()
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_train(config_path)
