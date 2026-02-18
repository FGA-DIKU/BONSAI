import logging
from os.path import join
import pandas as pd
import torch
import os

from corebehrt.constants.paths import (
    FOLDS_FILE,
    PREPARED_ALL_PATIENTS,
    XGBOOST_CFG,
    VOCABULARY_FILE,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main.helper.evaluate_finetune import compute_metrics
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.main.helper.evaluate_xgboost import xgb_inference_fold

CONFIG_PATH = "./corebehrt/configs/evaluate_xgboost.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate(finetune_cfg=XGBOOST_CFG)

    # Logger
    logger = logging.getLogger("evaluate")

    # Load data
    loaded_data = torch.load(join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False)
    vocab = torch.load(join(cfg.paths.test_data_dir, VOCABULARY_FILE))
    test_data = PatientDataset(loaded_data)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.folds_dir, FOLDS_FILE))
    check_for_overlap(folds, test_pids, logger)
    targets = test_data.get_outcomes()

    # Get predictions
    combined_df = pd.DataFrame(
        {
            "pid": test_pids,
            "target": targets,
        }
    )

    all_probas = []
    for n_fold in range(1, len(folds) + 1):
        logger.info(f"Evaluating fold {n_fold}/{len(folds)}")
        probas, fi_df = xgb_inference_fold(
            model_folder=cfg.paths.model,
            test_data=test_data,
            fold=n_fold,
            logger=logger,
            fi_cfg=cfg.get("return_feature_importance", None),
            vocab=vocab,
        )
        combined_df[f"fold_{n_fold}_probas"] = probas
        all_probas.append(probas)

        if fi_df is not None:
            os.makedirs(
                join(cfg.paths.predictions, "feature_importance"), exist_ok=True
            )
            fi_df.to_csv(
                join(cfg.paths.predictions, "feature_importance", f"fold_{n_fold}.csv"),
                index=False,
            )
            logger.info(f"Feature importance saved for fold {n_fold}")

    # Save predictions
    combined_df.to_csv(join(cfg.paths.predictions, "predictions.csv"), index=False)

    # Compute and log metrics
    compute_metrics(cfg, targets, all_probas, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
