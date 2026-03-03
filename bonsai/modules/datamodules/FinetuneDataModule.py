import lightning as L
import torch
from typing import Literal
from torch.utils.data import DataLoader
from bonsai.functional.collate import dynamic_padding
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
from bonsai.functional.features import get_background_length
from bonsai.functional.sampling import get_sampler


class FinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        path_train_data: list,
        path_val_data: list,
        vocabulary: list,
        train_labels: list,
        val_labels: list,
        test_labels: list,
        train_sampler,
        val_sampler,
        # train_transforms: Optional[Compose] = pretrain_CPU_train_transforms,
        # val_transforms: Optional[Compose] = pretrain_CPU_val_transforms,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.vocabulary = vocabulary
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.val_sampler = val_sampler
        self.train_sampler = train_sampler
        # self.train_transforms = train_transforms
        # self.val_transforms = val_transforms

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        train_data = torch.load(self.path_train_data)
        train_data = [
            sub
            for sub in train_data
            if sub["subject_id"].item() in self.train_labels["subject_id"].to_list()
        ]
        val_data = torch.load(self.path_val_data)
        val_data = [
            sub
            for sub in val_data
            if sub["subject_id"].item() in self.val_labels["subject_id"].to_list()
        ]

        background_length = get_background_length(
            df=train_data[0]["code"],
            vocabulary=self.vocabulary,
        )

        self.train_dataset = FinetuneDataset(
            train_data,
            vocabulary=self.vocabulary,
            background_tokens_per_patient=background_length,
            labels=self.train_labels,
        )
        self.val_dataset = FinetuneDataset(
            val_data,
            vocabulary=self.vocabulary,
            background_tokens_per_patient=background_length,
            labels=self.val_labels,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,  # Why is shuffle false?
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
            sampler=self.val_sampler,
        )
