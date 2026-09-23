import torch
from torch.utils.data import DataLoader

from bonsai.functional.collate import tte_collate_fn
from bonsai.functional.subject_data import filter_subject_data
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.datasets.TTEDataset import TTEDataset


class TTEDataModule(FinetuneDataModule):
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

        self.train_dataset = TTEDataset(
            train_data,
            outcomes=self.train_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )
        self.val_dataset = TTEDataset(
            val_data,
            outcomes=self.val_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )

    def setup_predict(self):
        if self.path_predict_data is None:
            raise ValueError("path_predict_data must be set before running predict.")
        predict_data = torch.load(self.path_predict_data)
        predict_data = [
            sub for sub in predict_data if sub["subject_id"] in self.predict_outcomes
        ]
        population_subject_ids = self.population["subject_id"].to_list()
        predict_data = filter_subject_data(predict_data, population_subject_ids)
        background_length = (predict_data[0]["segment"] == 0).sum()
        self.predict_dataset = TTEDataset(
            predict_data,
            outcomes=self.predict_outcomes,
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
            collate_fn=tte_collate_fn,
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
            collate_fn=tte_collate_fn,
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
            collate_fn=tte_collate_fn,
        )
