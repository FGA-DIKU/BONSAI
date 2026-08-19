from pathlib import Path

import hydra
import lightning as L
import polars as pl
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from transformers import ModernBertConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from bonsai.modules.callbacks.loss_plot import LossPlotCallback

from bonsai.functional.features import compute_abspos
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.outcomes import split_and_binarize_outcomes
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.sampling import get_sampler
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune
from bonsai.functional.outcomes import (
    resolve_duplicate_subject_outcomes,
    split_and_binarize_outcomes,
    print_outcome_split_summary,
)
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.features import compute_abspos
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.paths import get_config_path

OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True
)

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="training_runs")
    model_save_dir = logger.log_dir

    vocab = torch.load(cfg.paths.vocabulary)
    outcomes = pl.read_parquet(cfg.paths.outcome)
    outcomes = outcomes.with_columns(
        censor_abspos=compute_abspos(pl.col("censor_date"))
    )
    outcomes = resolve_duplicate_subject_outcomes(
        outcomes, cfg.outcomes.duplicate_subject_policy
    )
    train_outcomes, val_outcomes, predict_outcomes = split_and_binarize_outcomes(
        outcomes,
        train_key="train",
        val_key="tuning",
        test_key="held_out",
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.n_hours_end_include,
    )
    print_outcome_split_summary(
        {
            "train": train_outcomes,
            "tuning": val_outcomes,
            "held_out": predict_outcomes,
        }
    )

    train_labels = [outcome["label"] for outcome in train_outcomes]
    data_module = FinetuneDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_predict_data=cfg.paths.predict_split,
        path_population=cfg.paths.population,
        train_outcomes=train_outcomes,
        val_outcomes=val_outcomes,
        predict_outcomes=predict_outcomes,
        predict_token_id=vocab["[CLS]"],
        max_len=cfg.training.max_len,
        sampling_weight_fn=cfg.training.sampling_weight_fn,
    )

    model = instantiate(
        cfg.model,
        vocab_size=len(vocab),
        predict_token_id=vocab["[CLS]"],
    )

    lightning_module = FinetuneModule(
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        weight_decay=cfg.training.get("weight_decay", 0.0),
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        pos_weight=get_loss_weight(
            cfg.training.loss_weight_function,
            labels=train_labels,
        ),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor=cfg.training.eval_monitor_metric,
            mode="min",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
        LossPlotCallback(
            metrics_csv=Path(model_save_dir) / "metrics.csv",
            save_path=Path(model_save_dir) / "loss.png",
        ),
    ]
    if cfg.training.get("early_stopping_patience") is not None:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.eval_monitor_metric,
                mode="min",
                patience=cfg.training.early_stopping_patience,
            )
        )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=callbacks,
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

    if cfg.paths.predict_split is not None:
        predictions_output_path = Path(model_save_dir) / "test_predictions"
        lightning_module.predictions_output_path = predictions_output_path
        trainer.predict(
            model=lightning_module,
            datamodule=data_module,
            ckpt_path="best",
        )
        print(f"Saved predictions to {predictions_output_path}")


# TODO: Aggregate scores here, assuming test has been run after each training and test outputs some file.


if __name__ == "__main__":
    main()
