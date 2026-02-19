from typing import Optional
from bonsai.modules.create_data import create_and_save_features
from bonsai.modules.create_data import load_tokenize_and_save
from bonsai.modules.features.tokenizer import EHRTokenizer
import lightning as L
from pathlib import Path
import torch


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        logger: L.Logger,
        splits,
        path_data: Path,
        path_tokenized: Path,
        path_features: Path,
        path_vocab: Optional[Path] = None,
        agg_kwargs=None,
        exclude_regex=None,
        values_kwargs=None,
        tokenizer_kwargs=None,
    ):
        super().__init__()

        # Paths
        self.path_data = path_data
        self.path_tokenized = path_tokenized
        self.path_features = path_features

        # create_and_save_features kwargs
        self.splits = splits
        self.agg_kwargs = agg_kwargs        
        self.exclude_regex = exclude_regex
        self.values_kwargs = values_kwargs

        # Tokenizer kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self.logger = logger

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to be done only from a single process, like downloading data, tokenization, etc."""
        self.logger.info("BaseDataModule: prepare_data")
        self._init_folders()

        # Creating and saving features
        create_and_save_features(
            path_data=self.path_data,
            path_features=self.path_features,
            splits=self.splits,
            logger=self.logger,
            agg_kwargs=self.agg_kwargs,
            exclude_regex=self.exclude_regex,
            values_kwargs=self.values_kwargs,
        )
        self.logger.info("Finished creating and saving features")

        # Initializing tokenizer and tokenizing
        vocabulary = None
        if self.path_vocab is not None:
            self.logger.info(f"Loading vocabulary from {self.path_vocab}")
            vocabulary = torch.load(self.path_vocab, weights_only=True)
        tokenizer = EHRTokenizer(
            vocabulary=vocabulary,
            **(self.tokenizer_kwargs or {}),
        )

        assert self.splits[0] == "train", "First split must be 'train' to build vocabulary before tokenizing other splits"
        for split in self.splits:
            self.logger.info(f"Tokenizing {split}")
            load_tokenize_and_save(
                self.path_features,
                tokenizer,
                self.path_tokenized,
                split,
            )

    def _init_folders(self):
        self.path_tokenized.mkdir(parents=True, exist_ok=True)
        self.path_features.mkdir(parents=True, exist_ok=True)

    def setup(self, stage=None):
        """Use this method to do things that might need to be done on every process, like loading data, applying transforms, etc."""
        pass

    def train_dataloader(self):
        """Return the training dataloader."""
        pass

    def val_dataloader(self):
        """Return the validation dataloader."""
        pass

    def test_dataloader(self):
        """Return the test dataloader."""
        pass
