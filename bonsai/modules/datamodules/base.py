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
        logger,
        splits,
        path_data: str,
        path_tokenized: str,
        path_features: str,
        path_vocab: Optional[str] = None,
        exclude_regex=None,
        agg_kwargs: Optional[dict] = None,
        values_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        print("UNUSED:", kwargs)

        # Paths
        self.path_data = Path(path_data)
        self.path_tokenized = Path(path_tokenized)
        self.path_features = Path(path_features)
        self.path_vocab = Path(path_vocab) if path_vocab is not None else None

        # create_and_save_features kwargs
        self.splits = splits
        self.exclude_regex = exclude_regex
        self.agg_kwargs = agg_kwargs        
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
        torch.save(tokenizer.vocabulary, self.path_tokenized / "vocabulary.pt")  # save vocabulary

    def _init_folders(self):
        self.path_tokenized.mkdir(parents=True, exist_ok=True)
        self.path_features.mkdir(parents=True, exist_ok=True)

    def setup(self, stage=None):
        """Use this method to do things that might need to be done on every process, like loading data, applying transforms, etc."""
        pass
