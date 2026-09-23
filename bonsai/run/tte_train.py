from pathlib import Path

import hydra
import lightning as L
import polars as pl
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.outcomes import split_and_tte_outcomes
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.modules.datamodules.TTEDataModule import TTEDataModule
from bonsai.modules.lightningmodules.TTEModule import TTEModule
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
    train_outcomes, val_outcomes, predict_outcomes = split_and_tte_outcomes(
        outcomes,
        train_key="train",
        val_key="tuning",
        test_key="held_out",
        n_hours_end_include=cfg.labels.n_hours_end_include,
        end_of_followup=cfg.labels.end_of_followup,
    )

    data_module = TTEDataModule(
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
    )

    model = instantiate(
        cfg.model,
        vocab_size=len(vocab),
        predict_token_id=vocab["[CLS]"],
        prediction_horizons=cfg.labels.prediction_horizons,
    )

    lightning_module = TTEModule(
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        prediction_horizons=cfg.labels.prediction_horizons,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor=cfg.training.eval_monitor_metric,
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
