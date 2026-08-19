from typing import Any, List, Literal, Optional
import polars as pl
import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from bonsai.functional.collate import dynamic_padding
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
from bonsai.functional.subject_data import filter_subject_data
from bonsai.functional.sampling import get_sampler
from bonsai.functional.outcomes import expand_subjects_for_outcomes, outcome_subject_ids


class FinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_data: str,
        path_val_data: str,
        path_predict_data: str,
        path_population: str,
        batch_size: int,
        num_workers: int,
        predict_token_id: int,
        max_len: int,
        train_outcomes: List[dict],
        val_outcomes: List[dict],
        predict_outcomes: List[dict],
        sampling_weight_fn: Optional[Any] = None,
        train_sampler: Optional[WeightedRandomSampler] = None,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.path_predict_data = path_predict_data
        self.population = pl.read_csv(path_population)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_token_id = predict_token_id
        self.max_len = max_len

        self.train_outcomes = train_outcomes
        self.val_outcomes = val_outcomes
        self.predict_outcomes = predict_outcomes
        self.sampling_weight_fn = sampling_weight_fn
        self.train_sampler = train_sampler

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self):
        train_data = torch.load(self.path_train_data)
        val_data = torch.load(self.path_val_data)

        train_subject_ids = outcome_subject_ids(self.train_outcomes)
        val_subject_ids = outcome_subject_ids(self.val_outcomes)

        train_data = [
            sub for sub in train_data if sub["subject_id"] in train_subject_ids
        ]
        val_data = [sub for sub in val_data if sub["subject_id"] in val_subject_ids]

        population_subject_ids = self.population["subject_id"].to_list()
        train_data = filter_subject_data(train_data, population_subject_ids)
        val_data = filter_subject_data(val_data, population_subject_ids)

        train_subjects, train_outcome_rows = expand_subjects_for_outcomes(
            train_data, self.train_outcomes
        )
        val_subjects, val_outcome_rows = expand_subjects_for_outcomes(
            val_data, self.val_outcomes
        )

        # !!! Assumes background tokens ALWAYS exists AND same for all people !!!
        background_length = (train_subjects[0]["segment"] == 0).sum()

        self.train_dataset = FinetuneDataset(
            train_subjects,
            outcome_rows=train_outcome_rows,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )
        self.val_dataset = FinetuneDataset(
            val_subjects,
            outcome_rows=val_outcome_rows,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )

    def setup_predict(self):
        if self.path_predict_data is None:
            raise ValueError("path_predict_data must be set before running predict.")
        predict_data = torch.load(self.path_predict_data)
        predict_subject_ids = outcome_subject_ids(self.predict_outcomes)
        predict_data = [
            sub for sub in predict_data if sub["subject_id"] in predict_subject_ids
        ]
        population_subject_ids = self.population["subject_id"].to_list()
        predict_data = filter_subject_data(predict_data, population_subject_ids)
        predict_subjects, predict_outcome_rows = expand_subjects_for_outcomes(
            predict_data, self.predict_outcomes
        )
        background_length = (predict_subjects[0]["segment"] == 0).sum()
        self.predict_dataset = FinetuneDataset(
            predict_subjects,
            outcome_rows=predict_outcome_rows,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )

    def train_dataloader(self):
        # WeightedRandomSampler weights must match dataset index i -> train_dataset[i].
        # Labels derived from dict.values() do not match subject order in train_data; build
        # the sampler here after setup_fit.
        sampler: Optional[WeightedRandomSampler] = None
        shuffle = False
        if self.sampling_weight_fn is not None:
            labels = [outcome["label"] for outcome in self.train_dataset.outcome_rows]
            sampler = get_sampler(self.sampling_weight_fn, labels)
        elif self.train_sampler is not None:
            sampler = self.train_sampler
        else:
            shuffle = True

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=dynamic_padding,
            sampler=sampler,
            shuffle=shuffle,
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

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            shuffle=False,
            collate_fn=dynamic_padding,
        )
