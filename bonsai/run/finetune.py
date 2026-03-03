import hydra
import lightning as L
import torch

from omegaconf import DictConfig, OmegaConf
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune
from transformers import ModernBertConfig
import pandas as pd
from bonsai.functional.loss import get_loss_weight
from lightning.pytorch.callbacks import ModelCheckpoint
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.paths import get_config_path
from lightning.pytorch.loggers import CSVLogger
from dotenv import load_dotenv
from bonsai.functional.sampling import get_sampler

load_dotenv()


def merge_configs_and_drop_duplicate_keys(pretrain_cfg, finetune_cfg):
    keys_to_drop = [
        "vocab_size",
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "sparse_prediction",
    ]
    finetune_cfg_as_regular_dict = OmegaConf.to_container(
        finetune_cfg, resolve=True, throw_on_missing=True
    )
    model_cfg = pretrain_cfg | finetune_cfg_as_regular_dict
    for key in keys_to_drop:
        model_cfg.pop(key)
    return model_cfg


def binarize_labels(labels, n_hours_start_include, n_hours_end_include=None):
    time_delta_datetime = labels["outcome_date"] - labels["index_date"]
    time_delta_hours = time_delta_datetime.dt.days * 24
    outcomes_in_prediction_window = time_delta_hours > n_hours_start_include
    if n_hours_end_include is not None:
        outcomes_in_prediction_window = time_delta_hours < n_hours_end_include
    labels["label"] = outcomes_in_prediction_window.astype(int)
    return labels


def split_and_binarize_labels(
    labels,
    train_key,
    val_key,
    test_key,
    n_hours_start_include,
    n_hours_end_include=None,
):
    train_labels = labels[labels["split"] == train_key]
    train_labels = binarize_labels(
        train_labels, n_hours_start_include, n_hours_end_include
    )
    val_labels = labels[labels["split"] == val_key]
    val_labels = binarize_labels(val_labels, n_hours_start_include, n_hours_end_include)
    test_labels = labels[labels["split"] == test_key]
    test_labels = binarize_labels(
        test_labels, n_hours_start_include, n_hours_end_include
    )

    return train_labels, val_labels, test_labels


@hydra.main(
    config_path=get_config_path(),
    config_name="finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    model_save_dir = get_experiment_output_path()
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = merge_configs_and_drop_duplicate_keys(
        pretrain_cfg=ckpt["hyper_parameters"], finetune_cfg=cfg
    )
    vocab = torch.load(cfg.data.path_vocab)
    outcomes = pd.read_parquet(cfg.data.path_outcome)
    train_labels, val_labels, test_labels = split_and_binarize_labels(
        outcomes,
        train_key="train",
        val_key="tuning",
        test_key="held_out",
        n_hours_start_include=cfg.data.n_hours_start_include,
        n_hours_end_include=cfg.data.n_hours_end_include,
    )
    logger = CSVLogger(model_save_dir, name="training_log")

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
        path_train_data=cfg.data.path_train_split,
        path_val_data=cfg.data.path_val_split,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        vocabulary=vocab,
        train_sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn,
            labels=train_labels["label"],
            label_counts=train_labels["label"].value_counts(),
        ),
        val_sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn,
            labels=val_labels["label"],
            label_counts=val_labels["label"].value_counts(),
        ),
    )

    model = BonsaiFinetune(
        ModernBertConfig(
            **model_cfg,
            vocab_size=len(vocab),
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
            cfg.training.loss_weight_function,
            label_counts=train_labels["label"].value_counts(),
        ),
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


# TODO: Aggregate scores here, assuming test has been run after each training and test outputs some file.


if __name__ == "__main__":
    main()
