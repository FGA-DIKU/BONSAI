from typing import Literal, Dict, Optional
import polars as pl
import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from bonsai.functional.collate import dynamic_padding
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
from bonsai.functional.subject_data import filter_subject_data


class FinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_data: str,
        path_val_data: str,
        path_population: str,
        batch_size: int,
        num_workers: int,
        predict_token_id: int,
        max_len: int,
        train_outcomes: Dict[int, dict],
        val_outcomes: Dict[int, dict],
        test_outcomes: Dict[int, dict],
        train_sampler: Optional[WeightedRandomSampler] = None,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.population = pl.read_csv(path_population)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_token_id = predict_token_id
        self.max_len = max_len

        self.train_outcomes = train_outcomes
        self.val_outcomes = val_outcomes
        self.test_outcomes = test_outcomes
        self.train_sampler = train_sampler

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        train_data = torch.load(self.path_train_data)
        val_data = torch.load(self.path_val_data)

        train_data = [
            sub for sub in train_data if sub["subject_id"] in self.train_outcomes
        ]
        val_data = [sub for sub in val_data if sub["subject_id"] in self.val_outcomes]

        population_subject_ids = self.population["subject_id"].to_list()
        train_data = filter_subject_data(train_data, population_subject_ids)
        val_data = filter_subject_data(val_data, population_subject_ids)

        # !!! Assumes background tokens ALWAYS exists AND same for all people !!!
        background_length = (train_data[0]["segment"] == 0).sum()

        self.train_dataset = FinetuneDataset(
            train_data,
            outcomes=self.train_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )
        self.val_dataset = FinetuneDataset(
            val_data,
            outcomes=self.val_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=dynamic_padding,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,
            collate_fn=dynamic_padding,
        )
