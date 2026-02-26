import lightning as L
import torch
from typing import Literal
from torch.utils.data import DataLoader
from bonsai.functional.collate import dynamic_padding
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
from bonsai.functional.features import get_background_length


class FinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        vocabulary: list,
        sampler=None,
        # train_transforms: Optional[Compose] = pretrain_CPU_train_transforms,
        # val_transforms: Optional[Compose] = pretrain_CPU_val_transforms,
        # num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.vocabulary = vocabulary
        self.sampler = sampler
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
        self.train_data = torch.load(self.path_train_data)
        self.val_data = torch.load(self.path_val_data)

        background_length = get_background_length(
            concepts=self.train_split[0]["concepts"],
            vocabulary=self.vocabulary,
        )
        self.train_dataset = FinetuneDataset(
            self.train_split,
            vocabulary=self.vocabulary,
            background_length=background_length,
        )
        self.val_dataset = FinetuneDataset(
            self.val_split,
            vocabulary=self.vocabulary,
            background_length=background_length,
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
            sampler=self.sampler,
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
            sampler=self.sampler,
        )
