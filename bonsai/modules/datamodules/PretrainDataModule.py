import pandas as pd
from typing import List, Optional
import torch
from torch.utils.data import DataLoader
import lightning as L
from bonsai.functional.collate import dynamic_padding
from bonsai.functional.subject_data import filter_subject_data
from bonsai.modules.datasets.PretrainDataset import (
    MLMPretrainDataset,
    ARPretrainDataset,
)


class PretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_data: str,
        path_val_data: str,
        path_vocab: str,
        path_population: str,
        batch_size: int,
        num_workers: int,
        dataset_class: torch.utils.data.Dataset,
        masking_config: Optional[dict] = None,
        cutoff_date: Optional[dict] = None,
        max_len: int = 8192,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.vocabulary = torch.load(path_vocab)
        self.population = pd.read_csv(path_population)
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
        train_data = torch.load(self.path_train_data)
        val_data = torch.load(self.path_val_data)
        train_data = filter_subject_data(train_data, self.population["subject_id"])
        val_data = filter_subject_data(val_data, self.population["subject_id"])

        # !!! Assumes background tokens ALWAYS exists AND same for all people !!!
        background_length = (train_data[0]["segment"] == 0).sum()

        if issubclass(self.dataset_class, MLMPretrainDataset):
            assert self.masking_config is not None
            self.train_dataset = self.dataset_class(
                train_data,
                max_len=self.max_len,
                cutoff_date=self.cutoff_date,
                background_length=background_length,
                vocabulary=self.vocabulary,
                masking_select_ratio=self.masking_config.masking_select_ratio,
                masking_ratio=self.masking_config.masking_ratio,
                masking_random_ratio=self.masking_config.masking_random_ratio,
                masking_ignore_special_tokens=self.masking_config.masking_ignore_special_tokens,
            )
            self.val_dataset = self.dataset_class(
                val_data,
                max_len=self.max_len,
                cutoff_date=self.cutoff_date,
                background_length=background_length,
                vocabulary=self.vocabulary,
                masking_select_ratio=self.masking_config.masking_select_ratio,
                masking_ratio=self.masking_config.masking_ratio,
                masking_random_ratio=self.masking_config.masking_random_ratio,
                masking_ignore_special_tokens=self.masking_config.masking_ignore_special_tokens,
            )
        elif issubclass(self.dataset_class, ARPretrainDataset):
            self.train_dataset = self.dataset_class(
                train_data,
                self.max_len,
                background_length=background_length,
                cutoff_date=self.cutoff_date,
            )
            self.val_dataset = self.dataset_class(
                val_data,
                self.max_len,
                background_length=background_length,
                cutoff_date=self.cutoff_date,
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
            drop_last=False,
            shuffle=False,
            collate_fn=dynamic_padding,
        )

    def set_logger(self):
        if self.trainer:
            return self.trainer.logger
        import logging

        return logging.getLogger()
