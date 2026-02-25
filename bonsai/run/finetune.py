import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune
from transformers import ModernBertConfig
from functional import partial
import pandas as pd
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.sampling import get_sampler


@hydra.main(
    # config_path=get_config_path(), # TODO: make this more flexible to allow for different config paths
    config_path="bonsai/configs",
    config_name="finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    finetune_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    pretrain_cfg = ModernBertConfig.from_pretrained(cfg.checkpoint_path)
    cfg = pretrain_cfg | finetune_cfg

    vocabulary = pd.read_csv("vocabulary")
    labels = pd.read_csv("outcomes")
    label_counts = pd.Series(labels).value_counts()
    sampler = get_sampler(
        weight_fn=cfg.sample_weight_fn, labels=labels, label_counts=label_counts
    )
    pos_weight = get_loss_weight(cfg, label_counts=label_counts)

    train_split = ()
    val_split = ()

    data_module = partial(
        FinetuneDataModule,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        train_split=train_split,
        val_split=val_split,
        vocabulary=vocabulary,
        sampler=sampler,
    )

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

    lightning_module = partial(
        FinetuneModule,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        pos_weight=pos_weight,
    )

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
