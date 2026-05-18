import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
import xgboost as xgb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torchmetrics.classification import (
    BinarySensitivityAtSpecificity,
    BinarySpecificityAtSensitivity,
)

from bonsai.functional.features import compute_abspos
from bonsai.functional.outcomes import (
    outcomes_to_frame,
    resolve_duplicate_subject_outcomes,
    split_and_binarize_outcomes,
)
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.paths import get_config_path

load_dotenv()

RUNS_DIR = "training_runs"
METRICS_FILE = "metrics.csv"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
TOP_FEATURES_FILE = "top_features.csv"
CV_RESULTS_FILE = "cv_results.csv"
BEST_PARAMS_FILE = "best_params.json"
TOP_FEATURES = 20


def split_key_from_pt(split_path: str) -> str:
    return Path(split_path).stem.removeprefix("subject_data_")


def split_dir_from_pt(split_path: str) -> Path:
    return Path(split_path).parent / split_key_from_pt(split_path)


def id_to_token_name(vocab: dict) -> Dict[int, str]:
    return {token_id: name for name, token_id in vocab.items()}


def feature_importance_df(
    model: xgb.XGBClassifier,
    id_to_name: Dict[int, str],
) -> pd.DataFrame:
    scores = model.get_booster().get_score(importance_type="gain")
    rows = []
    for feature_key, importance in scores.items():
        token_id = int(feature_key.removeprefix("f"))
        rows.append(
            {
                "token_id": token_id,
                "code": id_to_name.get(token_id, f"<id_{token_id}>"),
                "importance": importance,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda df: df.index + 1)
    )


def log_top_features(
    importance_df: pd.DataFrame,
    top_k: int = TOP_FEATURES,
) -> None:
    top = importance_df.head(top_k)
    print(f"\nTop {len(top)} features by gain:")
    for row in top.itertuples(index=False):
        print(
            f"  {row.rank:3d}. {row.code} "
            f"(id={row.token_id}, gain={row.importance:.4f})"
        )


def token_ids_to_exclude(vocab: dict, code_names: List[str]) -> Set[int]:
    return {vocab[name] for name in code_names if name in vocab}


def load_subject_events(
    split_dir: Path,
    subject_ids: Set[int] | None = None,
) -> Dict[int, pd.DataFrame]:
    events_by_subject: Dict[int, pd.DataFrame] = {}
    for shard in split_dir.glob("*.parquet"):
        df = pd.read_parquet(shard, columns=["subject_id", "code", "abspos"])
        if subject_ids is not None:
            df = df[df["subject_id"].isin(subject_ids)]
        for subject_id, group in df.groupby("subject_id", sort=False):
            subject_id = int(subject_id)
            if subject_id in events_by_subject:
                events_by_subject[subject_id] = pd.concat(
                    [events_by_subject[subject_id], group]
                )
            else:
                events_by_subject[subject_id] = group
    for subject_id, group in events_by_subject.items():
        events_by_subject[subject_id] = group.sort_values("abspos")
    return events_by_subject


def censored_codes_for_subject(
    events: pd.DataFrame,
    censor_abspos,
) -> Set[int]:
    if pd.isnull(censor_abspos):
        truncated = events
    else:
        abspos = events["abspos"].to_numpy()
        truncated = events.iloc[: bisect_right(abspos, censor_abspos)]
    return {
        int(code)
        for code in truncated["code"].tolist()
    }


def multihot_encode_outcomes(
    labels: pl.DataFrame,
    events_by_subject: Dict[int, pd.DataFrame],
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    subject_ids = []
    for outcome in labels.to_dicts():
        subject_id = int(outcome["subject_id"])
        events = events_by_subject.get(subject_id)
        if events is None:
            codes: Set[int] = set()
        else:
            codes = censored_codes_for_subject(
                events, outcome["censor_abspos"]
            )
        row = np.zeros(n_features, dtype=np.float32)
        for code in codes:
            if 0 <= code < n_features:
                row[code] = 1.0
        rows.append(row)
        subject_ids.append(subject_id)
    return np.stack(rows), labels["label"].to_numpy(), np.array(subject_ids)


def build_split_matrix(
    split_dir: Path,
    labels: pl.DataFrame,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_ids = set(labels["subject_id"].to_list())
    events_by_subject = load_subject_events(split_dir, subject_ids)
    return multihot_encode_outcomes(
        labels, events_by_subject, n_features
    )


def get_versioned_run_dir(experiment_dir: Path, runs_dir_name: str) -> Path:
    runs_root = experiment_dir / runs_dir_name
    runs_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.split("_")[1])
        for path in runs_root.iterdir()
        if path.is_dir() and path.name.startswith("version_")
    ]
    version = max(existing, default=-1) + 1
    run_dir = runs_root / f"version_{version}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def compute_split_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y = torch.tensor(y_true, dtype=torch.int64)
    p = torch.tensor(prob, dtype=torch.float32)
    sens_at_spec, _ = BinarySensitivityAtSpecificity(min_specificity=0.85)(p, y)
    spec_at_sens, _ = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(p, y)
    return {
        "auc": roc_auc_score(y_true, prob),
        "sensitivity_at_specificity_0.85": float(sens_at_spec),
        "specificity_at_sensitivity_0.70": float(spec_at_sens),
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def log_split_metrics(split: str, metrics: dict) -> None:
    print(
        f"{split} AUC: {metrics['auc']:.4f} "
        f"(n={metrics['n']}, positives={metrics['n_positive']})"
    )
    print(
        f"{split} sensitivity @ specificity 0.85: "
        f"{metrics['sensitivity_at_specificity_0.85']:.4f}"
    )
    print(
        f"{split} specificity @ sensitivity 0.70: "
        f"{metrics['specificity_at_sensitivity_0.70']:.4f}"
    )


def save_predictions(
    path: Path,
    subject_ids: np.ndarray,
    prob: np.ndarray,
    labels: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "subject_id": subject_ids,
            "prob": prob,
            "label": labels,
        }
    ).to_csv(path, index=False)


def scale_pos_weight_from_labels(y_train: np.ndarray, cfg: DictConfig) -> float:
    scale_pos_weight = cfg.xgb.scale_pos_weight
    if scale_pos_weight is None:
        n_pos = max(int(y_train.sum()), 1)
        n_neg = max(int(len(y_train) - y_train.sum()), 1)
        scale_pos_weight = n_neg / n_pos
    return float(scale_pos_weight)


def build_classifier(
    cfg: DictConfig,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBClassifier:
    xgb_cfg = cfg.xgb
    classifier_kwargs = {
        "n_estimators": xgb_cfg.n_estimators,
        "max_depth": xgb_cfg.max_depth,
        "learning_rate": xgb_cfg.learning_rate,
        "subsample": xgb_cfg.subsample,
        "colsample_bytree": xgb_cfg.colsample_bytree,
        "min_child_weight": xgb_cfg.min_child_weight,
        "reg_alpha": xgb_cfg.reg_alpha,
        "reg_lambda": xgb_cfg.reg_lambda,
        "gamma": xgb_cfg.gamma,
        "scale_pos_weight": scale_pos_weight_from_labels(y_train, cfg),
        "eval_metric": xgb_cfg.eval_metric,
        "random_state": xgb_cfg.random_state,
    }
    if params:
        classifier_kwargs.update(params)
    return xgb.XGBClassifier(**classifier_kwargs)


def tune_hyperparameters(
    cfg: DictConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    tuning_cfg = cfg.tuning
    param_grid = OmegaConf.to_container(tuning_cfg.param_grid, resolve=True)
    cv = StratifiedKFold(
        n_splits=tuning_cfg.cv_folds,
        shuffle=True,
        random_state=cfg.xgb.random_state,
    )
    search = GridSearchCV(
        estimator=build_classifier(cfg, y_train),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=tuning_cfg.n_jobs,
        verbose=tuning_cfg.verbose,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def fit_classifier(
    cfg: DictConfig,
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    # XGBoost >= 2.1: early_stopping_rounds is a constructor param, not a fit() kwarg.
    if cfg.xgb.early_stopping_rounds is not None:
        model.set_params(early_stopping_rounds=cfg.xgb.early_stopping_rounds)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=cfg.xgb.verbose,
    )
    return model


@hydra.main(
    config_path=get_config_path(),
    config_name="xgb",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    vocab = torch.load(cfg.paths.vocabulary)
    n_features = max(vocab.values()) + 1

    outcomes = pl.read_parquet(cfg.paths.outcome)
    outcomes = outcomes.with_columns(
        censor_abspos=compute_abspos(pl.col("censor_date"))
    )
    outcomes = resolve_duplicate_subject_outcomes(
        outcomes, cfg.outcomes.duplicate_subject_policy
    )
    train_outcomes, val_outcomes, test_outcomes = split_and_binarize_outcomes(
        outcomes,
        train_key=split_key_from_pt(cfg.paths.train_split),
        val_key=split_key_from_pt(cfg.paths.val_split),
        test_key=split_key_from_pt(cfg.paths.test_split),
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.n_hours_end_include,
    )

    train_labels = outcomes_to_frame(train_outcomes)
    val_labels = outcomes_to_frame(val_outcomes)
    test_labels = outcomes_to_frame(test_outcomes)

    X_train, y_train, _ = build_split_matrix(
        split_dir_from_pt(cfg.paths.train_split),
        train_labels,
        n_features,
    )
    X_val, y_val, _ = build_split_matrix(
        split_dir_from_pt(cfg.paths.val_split),
        val_labels,
        n_features,
    )
    X_test, y_test, test_ids = build_split_matrix(
        split_dir_from_pt(cfg.paths.test_split),
        test_labels,
        n_features,
    )

    run_dir = get_versioned_run_dir(Path(get_experiment_output_path()), RUNS_DIR)

    best_params: Dict[str, Any] = {}
    if cfg.tuning.enabled:
        search = tune_hyperparameters(cfg, X_train, y_train)
        best_params = dict(search.best_params_)
        print(
            f"Grid search best CV ROC-AUC: {search.best_score_:.4f} "
            f"({cfg.tuning.cv_folds}-fold)"
        )
        print(f"Best params: {best_params}")
        pd.DataFrame(search.cv_results_).to_csv(run_dir / CV_RESULTS_FILE, index=False)
        with open(run_dir / BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"best_cv_roc_auc": search.best_score_, "params": best_params},
                f,
                indent=2,
            )

    model = build_classifier(cfg, y_train, params=best_params or None)
    model = fit_classifier(cfg, model, X_train, y_train, X_val, y_val)

    importance_df = feature_importance_df(model, id_to_token_name(vocab))
    log_top_features(importance_df)
    top_features_path = run_dir / TOP_FEATURES_FILE
    importance_df.head(TOP_FEATURES).to_csv(top_features_path, index=False)

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics_rows = []
    for split, y_true, prob in [
        ("train", y_train, train_prob),
        ("val", y_val, val_prob),
    ]:
        split_metrics = {"split": split, **compute_split_metrics(y_true, prob)}
        metrics_rows.append(split_metrics)
        log_split_metrics(split, split_metrics)

    metrics_path = run_dir / METRICS_FILE
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    test_predictions_path = run_dir / TEST_PREDICTIONS_FILE
    save_predictions(test_predictions_path, test_ids, test_prob, y_test)

    print(f"Saved run outputs to {run_dir}")
    print(f"  metrics: {metrics_path}")
    if cfg.tuning.enabled:
        print(f"  cv results: {run_dir / CV_RESULTS_FILE}")
        print(f"  best params: {run_dir / BEST_PARAMS_FILE}")
    print(f"  top features: {top_features_path}")
    print(f"  test predictions: {test_predictions_path}")


if __name__ == "__main__":
    main()
