from typing import Dict, List, Literal, Optional
import torch
from torch.utils.data import DataLoader
from bonsai.functional.collate import dynamic_padding
import lightning as L
from bonsai.modules.datasets.PretrainDataset import MLMPretrainDataset, ARPretrainDataset
from pathlib import Path
from bonsai.functional.subject_data import filter_subject_data

class PretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        logger,
        path_tokenized: str,
        path_vocab: str,
        batch_size: int,
        num_workers: int,
        pretrain_mode: Literal["MLM", "AR"],
        masking_kwargs: Optional[dict] = None,
        cohorts: Optional[Dict[str, list]] = None,
        cutoff_date: Optional[dict] = None,
        max_len: int = 8192,
    ):
        super().__init__()
        self.logger = logger
        self.path_tokenized = Path(path_tokenized)
        self.cohorts = cohorts
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.max_len = max_len
        self.cutoff_date = cutoff_date

        self.pretrain_mode = pretrain_mode
        self.masking_kwargs = masking_kwargs
        self.vocabulary = torch.load(path_vocab, weights_only=True) # TODO: Out of place, only used in mlm AND in LightningModule to do len(datamodule.vocab)

    def prepare_data(self) -> None:
        # TODO: This can optionally be done in setup to reduce start-up time
        self.subject_data = {}
        for split in self.path_tokenized.glob("subject_data_*.pt"):
            split_name = split.stem.split("_")[2]
            self.logger.info(f"Loading subject data for split: {split.name}")
            subject_data = torch.load(split, weights_only=False)

            self.subject_data[split_name] = subject_data
    

    def setup(self, stage: str):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        self.cohort_filtering(["train", "tuning"])

        if self.pretrain_mode == "MLM":
            if self.vocabulary is None:
                raise ValueError("vocabulary is required for MLM pretraining mode")
            self.train_dataset = MLMPretrainDataset(
                self.subject_data["train"],
                vocabulary=self.vocabulary,
                **(self.masking_kwargs or {}),
                max_len=self.max_len,
            )
            self.val_dataset = MLMPretrainDataset(
                self.subject_data["tuning"],
                vocabulary=self.vocabulary,
                **(self.masking_kwargs or {}),
                max_len=self.max_len,
            )
        elif self.pretrain_mode == "AR":
             self.train_dataset = ARPretrainDataset(self.subject_data["train"], self.max_len)
             self.val_dataset = ARPretrainDataset(self.subject_data["tuning"], self.max_len)

    def cohort_filtering(self, splits: List[str]):
        if self.cohorts is not None:
            for split in splits:
                if split in self.cohorts:
                    self.logger.info(f"Filtering {split} data to cohort: {self.cohorts[split]}")
                    self.subject_data[split] = filter_subject_data(self.subject_data[split], self.cohorts[split], self.logger)
                else:
                    self.logger.warning(f"No cohort specified for split {split}, skipping cohort filtering")

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

