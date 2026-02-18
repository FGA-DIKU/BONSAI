import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.networks.bonsai import BonsaiFinetune
from transformers import ModernBertConfig
from functional import partial
from corebehrt.modules.trainer.utils import get_loss_weight


@hydra.main(
    # config_path=get_config_path(), # TODO: make this more flexible to allow for different config paths
    config_path="./corebehrt/configs",
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    finetune_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    pretrain_cfg = ModernBertConfig.from_pretrained(cfg.checkpoint_path)
    cfg = pretrain_cfg | finetune_cfg

    # TODO: implement path config for BONSAI
    # DirectoryPreparer(cfg).setup_pretrain()

    # TODO: implement logging for BONSAI -> Move this to LightningModule

    # TODO: Move path to the LightningTrainer so it auto-loads checkpoint from path
    # restart_path = cfg.paths.get("restart_model")
    # if restart_path:
    #    cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # TODO: Implement DataModule here and instantiate datasets in there.
    data_module = partial()
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

    model = partial(
        BonsaiFinetune,
        ModernBertConfig(
            **cfg.model,
            vocab_size=len(data_module.train_dataset.vocabulary),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            sparse_prediction=True,
        ),
    )

    pos_weight = get_loss_weight(cfg, outcomes=data_module.get_outcomes())

    lightning_module = partial(
        FinetuneModule,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        pos_weight=pos_weight,
    )

    # TODO: Implement Lightning Trainer here and pass in model, optimizer, scheduler, datasets, etc.
    trainer = partial(
        L.Trainer,
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        callbacks=[],
        loggers=[],
        max_epochs=cfg.training.max_epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.training.precision,
    )

    for i in range(cfg.num_folds):
        data_module = data_module(split=i)
        model = model()
        lightning_module = lightning_module(model=model)
        trainer = trainer()

        trainer.fit(
            model=lightning_module,
            datamodule=data_module,
            ckpt_path="last",
        )

    # TODO: Aggregate scores here, assuming test has been run after each training and test outputs some file.


if __name__ == "__main__":
    main()
