from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from bonsai.functional.collate import dynamic_padding
import lightning as L
from bonsai.modules.datasets.PretrainDataset import (
    MLMPretrainDataset,
    ARPretrainDataset,
)
from pathlib import Path
from bonsai.functional.subject_data import filter_subject_data


class PretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_data: Path,
        path_val_data: Path,
        path_vocab: Path,
        batch_size: int,
        num_workers: int,
        dataset_class: torch.utils.data.Dataset,
        masking_config: Optional[dict] = None,
        cohorts: Optional[Dict[str, list]] = None,
        cutoff_date: Optional[dict] = None,
        max_len: int = 8192,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.vocabulary = torch.load(path_vocab)
        self.cohorts = cohorts
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.max_len = max_len
        self.cutoff_date = cutoff_date

        self.dataset_class = dataset_class
        self.masking_config = masking_config
        self.logger = self.set_logger()

    def setup(self, stage: str):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        self.train_data = torch.load(self.path_train_data)
        self.val_data = torch.load(self.path_val_data)
        self.train_data = self.cohort_filtering("train", self.train_data)
        self.val_data = self.cohort_filtering("tuning", self.val_data)

        if issubclass(self.dataset_class, MLMPretrainDataset):
            self.train_dataset = self.dataset_class(
                self.train_data,
                cutoff_date=self.cutoff_date,
                vocabulary=self.vocabulary,
                masking_select_ratio=self.masking_config.masking_select_ratio,
                masking_ratio=self.masking_config.masking_ratio,
                masking_random_ratio=self.masking_config.masking_random_ratio,
                masking_ignore_special_tokens=self.masking_config.masking_ignore_special_tokens,
                max_len=self.max_len,
            )
            self.val_dataset = self.dataset_class(
                self.val_data,
                cutoff_date=self.cutoff_date,
                vocabulary=self.vocabulary,
                masking_select_ratio=self.masking_config.masking_select_ratio,
                masking_ratio=self.masking_config.masking_ratio,
                masking_random_ratio=self.masking_config.masking_random_ratio,
                masking_ignore_special_tokens=self.masking_config.masking_ignore_special_tokens,
                max_len=self.max_len,
            )
        elif issubclass(self.dataset_class, ARPretrainDataset):
            self.train_dataset = self.dataset_class(self.train_data, self.max_len, cutoff_date=self.cutoff_date)
            self.val_dataset = self.dataset_class(self.val_data, self.max_len, cutoff_date=self.cutoff_date)

    def cohort_filtering(self, split: str, data: List[dict]) -> List[dict]:
        if self.cohorts is not None and split in self.cohorts:
            self.logger.info(f"Filtering {split} subject_data to cohort {split}")
            return filter_subject_data(data, self.cohorts[split])
        else:
            self.logger.warning(f"No cohort specified for split {split}, skipping cohort filtering")
            return data

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            collate_fn=dynamic_padding,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,
            collate_fn=dynamic_padding,
        )

    def set_logger(self):
        if self.trainer:
            return self.trainer.logger
        import logging

        return logging.getLogger()
