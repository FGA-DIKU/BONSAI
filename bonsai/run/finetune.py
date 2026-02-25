import hydra
import lightning as L
from omegaconf import DictConfig
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune
from transformers import ModernBertConfig
import pandas as pd
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.sampling import get_sampler
from lightning.pytorch.callbacks import ModelCheckpoint
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.paths import get_config_path
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    model_save_dir = get_experiment_output_path()
    HFConfig = ModernBertConfig.from_pretrained(cfg.checkpoint_path)
    HFConfig = HFConfig.update(cfg)

    vocabulary = ()
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 1]
    label_counts = pd.Series(labels).value_counts()

    logger = CSVLogger(model_save_dir, name="training_log")

    train_split = ()
    val_split = ()

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

    data_module = FinetuneDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        train_split=train_split,
        val_split=val_split,
        vocabulary=vocabulary,
        sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn,
            labels=labels,
            label_counts=label_counts,
        ),
    )

    model = BonsaiFinetune(
        ModernBertConfig(
            **cfg.model,
            vocab_size=len(vocabulary),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            sparse_prediction=True,
        ),
    )

    lightning_module = FinetuneModule(
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        pos_weight=get_loss_weight(
            cfg.training.loss_weight_function, label_counts=label_counts
        ),
    )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
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


# TODO: Aggregate scores here, assuming test has been run after each training and test outputs some file.


if __name__ == "__main__":
    main()
