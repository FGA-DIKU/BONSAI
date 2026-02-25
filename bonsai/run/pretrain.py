import hydra
import lightning as L
from omegaconf import DictConfig
from bonsai.modules.lightningmodules.PretrainModule import PretrainModule
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.paths import get_config_path
from transformers import ModernBertConfig
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from bonsai.functional.pathing import get_experiment_output_path
from dotenv import load_dotenv
from hydra.utils import get_class

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="training_runs")
    model_save_dir = logger.log_dir

    vocab = (
        "/Users/zcr545/Desktop/Projects/repos/BONSAI/outputs/tokenized/vocabulary.pt"
    )
    train_data = "/Users/zcr545/Desktop/Projects/repos/BONSAI/outputs/tokenized/subject_data_train.pt"
    val_data = "/Users/zcr545/Desktop/Projects/repos/BONSAI/outputs/tokenized/subject_data_tuning.pt"

    best_ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        every_n_epochs=cfg.training.ckpt_every_n_epoch,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )

    data_module = PretrainDataModule(
        path_train_data=train_data,
        path_val_data=val_data,
        path_vocab=vocab,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        dataset_class=get_class(cfg.data.dataset_class),
        masking_config=cfg.training.masking,
    )

    model = BonsaiPretrain(
        ModernBertConfig(
            **cfg.model,
            vocab_size=len(data_module.vocabulary),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            sparse_prediction=True,
        )
    )

    lightning_module = PretrainModule(
        model=model,
        compile_mode=cfg.hardware.compile_mode,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
    )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[last_ckpt_callback, best_ckpt_callback],
        logger=[logger],
        max_epochs=cfg.training.epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
    )

    trainer.fit(
        model=lightning_module,
        datamodule=data_module,
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
