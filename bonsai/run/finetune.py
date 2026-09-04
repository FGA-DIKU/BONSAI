from pathlib import Path

import hydra
import lightning as L
import polars as pl
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.features import compute_abspos
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.outcomes import split_and_binarize_outcomes
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.sampling import get_sampler
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
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
    print(
        f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n"
    )

    logger = CSVLogger(get_experiment_output_path(), name="training_runs")
    model_save_dir = logger.log_dir

    ckpt = torch.load(cfg.pretrain_path, map_location="cpu", weights_only=False)
    pretrain_cfg = ckpt["hyper_parameters"]

    vocab = torch.load(cfg.paths.vocabulary)
    outcomes = pl.read_parquet(cfg.paths.outcome)
    outcomes = outcomes.with_columns(
        censor_abspos=compute_abspos(pl.col("censor_date"))
    )
    train_outcomes, val_outcomes, predict_outcomes = split_and_binarize_outcomes(
        outcomes,
        train_key="train",
        val_key="tuning",
        test_key="held_out",
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.n_hours_end_include,
    )

    train_labels = [v["label"] for v in train_outcomes.values()]
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
        train_sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn, labels=train_labels
        ),
    )

    model = instantiate(
        cfg.model,
        vocab_size=len(vocab),
        max_seqlen=pretrain_cfg["max_seqlen"],
        hidden_size=pretrain_cfg["hidden_size"],
        num_layers=pretrain_cfg["num_layers"],
        num_attention_heads=pretrain_cfg["num_attention_heads"],
        bias=pretrain_cfg["bias"],
        attn_type=pretrain_cfg["attn_type"],
        abspos_encoding=pretrain_cfg.get("abspos_encoding", "scaled_time2vec"),
        predict_token_id=vocab["[CLS]"],
    )

    lightning_module = FinetuneModule.load_from_checkpoint(
        cfg.pretrain_path,
        strict=False,
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        pos_weight=get_loss_weight(
            cfg.training.loss_weight_function,
            labels=train_labels,
        ),
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
