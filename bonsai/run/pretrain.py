import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.lightningmodules.PretrainModule import PretrainModule


@hydra.main(
    # config_path=get_config_path(), # TODO: make this more flexible to allow for different config paths
    config_path="./corebehrt/configs",
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # TODO: implement path config for BONSAI
    # DirectoryPreparer(cfg).setup_pretrain()

    # TODO: implement logging for BONSAI
    # logger = logging.getLogger("pretrain")

    # TODO: Move path to the LightningTrainer so it auto-loads checkpoint from path
    # restart_path = cfg.paths.get("restart_model")
    # if restart_path:
    #    cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # TODO: Implement DataModule here and instantiate datasets in there.
    # data_module = (TODO)
    # train_data = PatientDataset(
    #    torch.load(join(cfg.paths.prepared_data, PREPARED_TRAIN_PATIENTS))
    # )
    # val_data = PatientDataset(
    #    torch.load(join(cfg.paths.prepared_data, PREPARED_VAL_PATIENTS))
    # )
    # vocab = load_vocabulary(cfg.paths.prepared_data)
    #     #train_dataset = MLMDataset(train_data.patients, vocab, **cfg.data.dataset)
    # val_dataset = MLMDataset(val_data.patients, vocab, **cfg.data.dataset)

    # TODO: Load Model here
    # model = initializer.initialize_pretrain_model(train_dataset)

    # TODO: Implement LightningModule here
    lightning_module = PretrainModule(
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
    )

    # TODO: Implement Lightning Trainer here and pass in model, optimizer, scheduler, datasets, etc.
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        callbacks=[],
        loggers=[],
        max_epochs=cfg.training.max_epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.training.precision,
    )

    trainer.fit(
        model=lightning_module,
        datamodule=data_module,
        ckpt_path="last",
    )

    # trainer = EHRTrainer(
    #    model=model,
    #    optimizer=optimizer,
    #    scheduler=scheduler,
    #    train_dataset=train_dataset,
    #    val_dataset=val_dataset,
    #    args=cfg.trainer_args,
    #    metrics=cfg.metrics,
    #    cfg=cfg,
    #    logger=logger,
    #    last_epoch=epoch,
    # )
    # logger.info("Start training")
    # trainer.train()
    # logger.info("Done")
