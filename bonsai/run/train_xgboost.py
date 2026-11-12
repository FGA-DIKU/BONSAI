"""Train an XGBoost baseline using BONSAI's train/tuning/held-out splits."""

from pathlib import Path

import hydra
import polars as pl
import torch
import xgboost as xgb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from bonsai.functional.outcomes import split_and_binarize_outcomes
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.subject_data import filter_subject_data
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.modules.datasets.XGBoostDataset import XGBoostDataset
from bonsai.paths import get_config_path

OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True
)

load_dotenv()


def load_dataset(path, outcomes, population, vocabulary, feature_mode):
    subjects = torch.load(path)
    subjects = [s for s in subjects if s["subject_id"] in outcomes]
    subjects = filter_subject_data(subjects, population)
    return XGBoostDataset(subjects, outcomes, vocabulary, feature_mode)


@hydra.main(
    config_path=get_config_path(),
    config_name="xgboost",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(
        f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n"
    )
    save_dir = Path(get_experiment_output_path())
    csv_logger = CSVLogger(save_dir, name=None, version=None)
    tb_logger = TensorBoardLogger(save_dir, name=None, version=csv_logger.version)

    vocabulary = torch.load(cfg.paths.vocabulary)
    outcomes = pl.read_parquet(cfg.paths.outcome)
    train_outcomes, val_outcomes, predict_outcomes = split_and_binarize_outcomes(
        outcomes,
        train_key="train",
        val_key="tuning",
        test_key="held_out",
        start_include=cfg.labels.start_include,
        end_include=cfg.labels.end_include,
    )
    population = pl.read_csv(cfg.paths.population)["subject_id"].to_list()

    train_dataset = load_dataset(
        cfg.paths.train_split, train_outcomes, population, vocabulary, cfg.features.mode
    )
    val_dataset = load_dataset(
        cfg.paths.val_split, val_outcomes, population, vocabulary, cfg.features.mode
    )
    X_train, y_train, _ = train_dataset.to_matrix()
    X_val, y_val, _ = val_dataset.to_matrix()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals_result = {}
    model = xgb.train(
        params=OmegaConf.to_container(cfg.model, resolve=True),
        dtrain=dtrain,
        num_boost_round=cfg.training.num_boost_round,
        evals=[(dtrain, "train"), (dval, "tuning")],
        evals_result=evals_result,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=cfg.training.early_stopping_rounds, save_best=True
            )
        ],
        verbose_eval=cfg.training.verbose_eval,
    )
    model.save_model(save_dir / "best.json")
    pl.DataFrame({"feature": train_dataset.feature_names}).write_csv(
        save_dir / "feature_names.csv"
    )

    metric = cfg.model.eval_metric
    for step, (train_score, val_score) in enumerate(
        zip(evals_result["train"][metric], evals_result["tuning"][metric])
    ):
        metrics = {f"train/{metric}": train_score, f"val/{metric}": val_score}
        csv_logger.log_metrics(metrics, step=step)
        tb_logger.log_metrics(metrics, step=step)

    importance = model.get_score(importance_type="gain")
    pl.DataFrame(
        {
            "feature": [train_dataset.feature_names[int(k[1:])] for k in importance],
            "gain": list(importance.values()),
        }
    ).write_csv(save_dir / "feature_importance.csv")

    if cfg.paths.predict_split is not None:
        predict_dataset = load_dataset(
            cfg.paths.predict_split,
            predict_outcomes,
            population,
            vocabulary,
            cfg.features.mode,
        )
        X_test, y_test, subject_ids = predict_dataset.to_matrix()
        probabilities = model.predict(xgb.DMatrix(X_test))
        predictions_dir = save_dir / "test_predictions"
        predictions_dir.mkdir(exist_ok=True)
        pl.DataFrame(
            {"subject_id": subject_ids, "prob": probabilities, "label": y_test}
        ).write_csv(predictions_dir / "predictions.csv")
        predict_metrics = {
            "predict/AUROC": roc_auc_score(y_test, probabilities),
            "predict/AveragePrecision": average_precision_score(y_test, probabilities),
        }
        pl.DataFrame([predict_metrics]).write_csv(
            predictions_dir / "predict_metrics.csv"
        )
        csv_logger.log_metrics(predict_metrics, step=model.best_iteration)
        tb_logger.log_metrics(predict_metrics, step=model.best_iteration)
        print(f"Saved predictions to {predictions_dir}")

    csv_logger.save()
    tb_logger.save()


if __name__ == "__main__":
    main()
