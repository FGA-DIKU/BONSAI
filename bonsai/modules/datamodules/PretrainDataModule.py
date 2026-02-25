from typing import Dict, List, Literal, Optional
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
        path_train_data: str,
        path_val_data: str,
        path_vocab: str,
        batch_size: int,
        num_workers: int,
        dataset_class: torch.utils.data.Dataset,
        masking_kwargs: Optional[dict] = None,
        cohorts: Optional[Dict[str, list]] = None,
        cutoff_date: Optional[dict] = None,
        max_len: int = 8192,
    ):
        super().__init__()
        self.path_train_data = Path(path_train_data)
        self.path_val_data = Path(path_val_data)
        self.vocabulary = torch.load(path_vocab)
        self.cohorts = cohorts
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.max_len = max_len
        self.cutoff_date = cutoff_date

        self.dataset_class = dataset_class
        self.masking_kwargs = masking_kwargs
        self.logger = self.set_logger()

    def setup(self, stage: str):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def log(self, message):
        if self.trainer:
            self.trainer.logger.info()

    def setup_fit(self):
        self.train_data = torch.load(self.path_train_data)
        self.val_data = torch.load(self.path_val_data)
        self.cohort_filtering(["train", "tuning"])

        if issubclass(self.dataset_class, MLMPretrainDataset):
            self.train_dataset = self.dataset_class(
                self.train_data,
                vocabulary=self.vocabulary,
                **(self.masking_kwargs or {}),
                max_len=self.max_len,
            )
            self.val_dataset = self.dataset_class(
                self.val_data,
                vocabulary=self.vocabulary,
                **(self.masking_kwargs or {}),
                max_len=self.max_len,
            )
        elif issubclass(self.dataset_class, ARPretrainDataset):
            self.train_dataset = self.dataset_class(self.train_data, self.max_len)
            self.val_dataset = self.dataset_class(self.val_data, self.max_len)

    def cohort_filtering(self, splits: List[str]):
        # This function is a bit cryptic, and I feel like it could be a bit simpler
        if self.cohorts is not None:
            for split in splits:
                if split in self.cohorts:
                    self.logger.info(
                        f"Filtering {split} data to cohort: {self.cohorts[split]}"
                    )
                    self.subject_data[split] = filter_subject_data(
                        self.subject_data[split], self.cohorts[split]
                    )
                else:
                    self.logger.warning(
                        f"No cohort specified for split {split}, skipping cohort filtering"
                    )

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
