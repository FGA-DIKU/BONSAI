import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.lightningmodules.PretrainModule import PretrainModule
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.paths import get_config_path
from transformers import ModernBertConfig
from lightning.pytorch.loggers import CSVLogger


@hydra.main(
    config_path=get_config_path(),
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(cfg)
    # TODO: implement path configuration for BONSAI
    # DirectoryPreparer(cfg).setup_pretrain()

    logger = CSVLogger("logs", name="test_experiment")
    # TODO: Implement instantiating transforms here
    # TODO: Instantiate callbacks and loggers here and pass to trainer

    # TODO: Move path to the LightningTrainer so it auto-loads checkpoint from path
    # restart_path = cfg.paths.get("restart_model")
    # if restart_path:
    #    cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # TODO: Implement DataModule here and instantiate datasets in there.
    ######  TEMP SOLUTION
    import torch
    import bonsai
    import os

    base_path = os.path.split(bonsai.__path__[0])[0]

    vocab = torch.load(
        os.path.join(base_path, "outputs/pretraining/processed_data/vocabulary.pt")
    )
    train_data = torch.load(
        os.path.join(
            base_path, "outputs/pretraining/processed_data/DICTpatients_train.pt"
        )
    )
    ###### TEMP SOLUTION

    data_module = PretrainDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        train_split=train_data,
        val_split=train_data,
        vocabulary=vocab,
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
        callbacks=[],
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
