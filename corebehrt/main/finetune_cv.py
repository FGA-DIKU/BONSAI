import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    TEST_PIDS_FILE,
    PREPARED_ALL_PATIENTS,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import cv_loop
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_and_save_scores_mean_std,
)
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/finetune.yaml"


def main_finetune(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_finetune()

    # Logger
    logger = logging.getLogger("finetune_cv")

    loaded_data = torch.load(
        join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS), weights_only=False
    )
    data = PatientDataset(loaded_data)

    test_data = PatientDataset([])

    # Initialize test and train/val pid lists
    test_pids = []
    train_val_pids = data.get_pids()

    # If evaluation is desired, then:
    #   - If test_pids are present in the prepared data directory, use them
    if cfg.get("evaluate", False):
        if os.path.exists(join(cfg.paths.prepared_data, TEST_PIDS_FILE)):
            test_pids = torch.load(
                join(cfg.paths.prepared_data, TEST_PIDS_FILE), weights_only=False
            )
            test_data = data.filter_by_pids(test_pids)
    train_val_data = data.filter_by_pids(train_val_pids)

    # Use folds from prepared data
    folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
    folds = torch.load(folds_path, weights_only=False)
    check_for_overlap(folds, test_pids, logger)
    n_folds = len(folds)
    logger.info(f"Using {n_folds} predefined folds")

    cv_loop(
        cfg,
        logger,
        cfg.paths.model,
        train_val_data,
        folds,
        test_data,
    )

    compute_and_save_scores_mean_std(n_folds, cfg.paths.model, mode="val")
    if len(test_data) > 0:
        compute_and_save_scores_mean_std(n_folds, cfg.paths.model, mode="test")

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
