import pandas as pd
import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import ModernBertConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.paths import get_config_path
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune
from bonsai.functional.outcomes import split_and_binarize_outcomes
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.sampling import get_sampler
from bonsai.functional.features import compute_abspos
from hydra.core.plugins import Plugins
from bonsai.modules.hydra.plugins import ExampleSearchpathPlugin

load_dotenv()
Plugins.instance().register(ExampleSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="training_runs")
    model_save_dir = logger.log_dir

    model_cfg = cfg.model

    vocab = torch.load(cfg.paths.vocabulary)
    outcomes = pd.read_parquet(cfg.paths.outcome)
    outcomes["censor_abspos"] = compute_abspos(outcomes["censor_date"])
    train_outcomes, val_outcomes, test_outcomes = split_and_binarize_outcomes(
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
        path_population=cfg.paths.population,
        train_outcomes=train_outcomes,
        val_outcomes=val_outcomes,
        test_outcomes=test_outcomes,
        predict_token_id=vocab["[CLS]"],
        train_sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn, labels=train_labels
        ),
    )

    model = BonsaiFinetune(
        ModernBertConfig(
            **model_cfg,
            vocab_size=len(vocab),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
        ),
    )

    lightning_module = FinetuneModule(
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


# TODO: Aggregate scores here, assuming test has been run after each training and test outputs some file.


if __name__ == "__main__":
    main()
