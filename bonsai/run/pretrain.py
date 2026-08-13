import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.modules.lightningmodules.PretrainModule import (
    PretrainModule,
)
from bonsai.paths import get_config_path

OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True
)

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(
        f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n"
    )

    logger = CSVLogger(get_experiment_output_path(), name=None, version=0)
    model_save_dir = logger.log_dir

    data_module = PretrainDataModule(
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_vocab=cfg.paths.vocab,
        path_population=cfg.paths.population,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        dataset_class=get_class(cfg.paths.dataset_class),
        masking_config=cfg.training.get("masking"),
        cutoff_date=cfg.training.cutoff_date,
        max_len=cfg.training.max_len,
    )

    model = instantiate(cfg.model, vocab_size=len(data_module.vocabulary))

    lightning_module = instantiate(
        cfg.lightning_module,
        model=model,
        compile_mode=cfg.hardware.compile_mode,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        loss_fn=cfg.training.loss_fn,
        loss_params=cfg.training.get("loss_params", {}),
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[ckpt_callback],
        logger=[logger],
        max_epochs=cfg.training.epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
    )

    trainer.fit(
        model=lightning_module,
        datamodule=data_module,
        ckpt_path=cfg.paths.ckpt_path,
    )


if __name__ == "__main__":
    main()
