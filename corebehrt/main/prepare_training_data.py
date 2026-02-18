"""Prepare data for training. Run create_data.yaml first to create the dataset and vocabulary."""

import logging
import torch
from os.path import join
import os

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.functional.features.split import split_pids_into_train_val
from corebehrt.main.helper.pretrain import (
    load_train_val_split,
    get_splits_path,
)
from corebehrt.functional.io_operations.save import save_pids_splits
from corebehrt.constants.paths import FOLDS_FILE, TEST_PIDS_FILE
from corebehrt.modules.preparation.dataset import PatientDataset

CONFIG_PATH = "./corebehrt/configs/prepare_pretrain.yaml"


def main_prepare_data(config_path):
    cfg = load_config(config_path)

    if cfg.data.type == "pretrain":
        # Setup directories
        DirectoryPreparer(cfg).setup_prepare_pretrain()
        logger = logging.getLogger("prepare pretrain data")
        logger.info("Preparing pretrain data")
        # Prepare data
        data = DatasetPreparer(cfg).prepare_pretrain_data(save_data=False)

        # Splitting data
        if cfg.data.get("predefined_splits", False):
            splits_path = get_splits_path(cfg.paths)
            train_data, val_data = load_train_val_split(data, splits_path)
        else:
            train_data, val_data = split_pids_into_train_val(
                data, cfg.data.get("val_ratio", 0.2)
            )
        save_pids_splits(train_data, val_data, cfg.paths.prepared_data)
        train_data.save(cfg.paths.prepared_data, suffix="_train")
        val_data.save(cfg.paths.prepared_data, suffix="_val")

    elif cfg.data.type == "finetune":
        # Setup directories
        DirectoryPreparer(cfg).setup_prepare_finetune()
        logger = logging.getLogger("prepare finetune data")
        logger.info("Preparing finetune data")
        # Prepare data
        _ = DatasetPreparer(cfg).prepare_finetune_data(mode="tuning")

        # Save splits from cohort selection
        folds_path = get_splits_path(cfg.paths)
        folds = torch.load(folds_path, weights_only=False)
        torch.save(folds, join(cfg.paths.prepared_data, FOLDS_FILE))
        test_pids_file = join(cfg.paths.cohort, TEST_PIDS_FILE)
        if os.path.exists(test_pids_file):
            test_pids = torch.load(test_pids_file, weights_only=False)
            torch.save(test_pids, join(cfg.paths.prepared_data, TEST_PIDS_FILE))

    elif cfg.data.type == "test":
        config_name = os.path.basename(config_path)
        DirectoryPreparer(cfg).setup_prepare_finetune(name=config_name)
        logger = logging.getLogger("prepare test data")
        logger.info("Preparing test data")
        # Prepare data
        combined_data = []
        for mode in cfg.data.get("test_modes", ["held_out"]):
            data = DatasetPreparer(cfg).prepare_finetune_data(mode=mode)
            combined_data.append(data)

        combined_patient_data = PatientDataset.combine_datasets(combined_data)
        combined_patient_data.save(cfg.paths.prepared_data)

    else:
        raise ValueError(f"Unsupported data type: {cfg.data.type}")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_prepare_data(config_path)
