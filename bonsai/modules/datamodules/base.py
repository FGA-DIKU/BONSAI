from typing import List, Optional
from bonsai.modules.create_data import create_features_and_tokenize
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer
import lightning as L
from pathlib import Path
import torch


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        logger,
        splits: List[str],
        path_data: str,
        path_tokenized: str,
        path_features: str,
        path_vocab: Optional[str] = None,
        exclude_regex=None,
        tokenizer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        print("UNUSED:", kwargs) # TODO: remove this after confirming that all kwargs are accounted for in the signature

        # Paths
        self.path_data = Path(path_data)
        self.path_tokenized = Path(path_tokenized)
        self.path_features = Path(path_features)

        # prepare_data kwargs
        self.splits = splits
        self.exclude_regex = exclude_regex

        # Initializing tokenizer
        vocabulary = None
        if path_vocab is not None:
            self.logger.info(f"Loading vocabulary from {path_vocab}")
            vocabulary = torch.load(path_vocab, weights_only=True)
        self.tokenizer = EHRTokenizer(
            vocabulary=vocabulary,
            **(tokenizer_kwargs or {}),
        )

        self.logger = logger

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to be done only from a single process, like downloading data, tokenization, etc."""
        self.logger.info("BaseDataModule: prepare_data")

        assert self.splits[0] == "train", "First split must be 'train' to build vocabulary before tokenizing other splits"
        for split in self.splits:
            self.logger.info(f"prepare_data: {split}")
            create_features_and_tokenize(
                split=split,
                path_data=self.path_data,
                path_tokenized=self.path_tokenized,
                logger=self.logger,
                tokenizer=self.tokenizer,
                exclude_regex=self.exclude_regex,
            )
            self.tokenizer.freeze_vocabulary()  # freeze after first split (train) to prevent data leakage
        torch.save(self.tokenizer.vocabulary, self.path_tokenized / "vocabulary.pt")  # save vocabulary

    def setup(self, stage=None):
        """Use this method to do things that might need to be done on every process, like loading data, applying transforms, etc."""
        pass
