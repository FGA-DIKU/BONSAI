import torch
import pandas as pd
import lightning as L
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from bonsai.functional.features import compute_abspos
from bonsai.modules.create_data import create_features_and_tokenize
from bonsai.functional.io_ops import load_concept
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer



class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        logger,
        splits: List[str],
        path_data: str,
        path_tokenized: str,
        path_features: str,
        path_cohort: Optional[str] = None,
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

        self.cohort = torch.load(Path(path_cohort) / "pids.pt", weights_only=False) if path_cohort is not None else None
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
        self.subject_data = prepare_training_format(
            splits=self.splits,
            path_tokenized=self.path_tokenized,
            cohort=self.cohort,
            cutoff_date=None,  # TODO: add cutoff date handling
            logger=self.logger,
        )
        return self.subject_data

def prepare_training_format(splits: List[str], path_tokenized: Path, cohort: List[int], cutoff_date: Optional[Dict], logger):
    """Load tokenized data and prepare it in the format needed for training."""
    prepared_data = {}
    for split in splits:
        split_path = path_tokenized / split
        all_tokenized = []
        for shard in split_path.glob("*.parquet"):
            logger.info(f"Preparing training format: {split}/{shard.name}")
            tokenized_data = pd.read_parquet(shard)

            # Filter subject_ids
            if cohort is not None:
                pre = tokenized_data["subject_id"].nunique()
                tokenized_data = tokenized_data[tokenized_data["subject_id"].isin(cohort)]
                post = tokenized_data["subject_id"].nunique()
                logger.info(f"Filtered subject_ids for {split}/{shard.name}: {pre} -> {post}")

            # Cutoff data
            if cutoff_date is not None:
                pre = len(tokenized_data)
                tokenized_data = cutoff_data(tokenized_data, cutoff_date)
                post = len(tokenized_data)
                logger.info(f"Cutoff data for {split}/{shard.name}: {pre} -> {post} rows")

            # Convert to training format
            for subject_id, group in tokenized_data.groupby("subject_id", sort=False):
                all_tokenized.append({
                    "subject_id": subject_id,
                    "concepts": torch.tensor(group["code"].tolist()),
                    "abspos": torch.tensor(group["abspos"].tolist()),
                    "segments": torch.tensor(group["segment"].tolist()),
                    "ages": torch.tensor(group["age"].tolist()),
                })

        prepared_data[split] = all_tokenized
    return prepared_data

def cutoff_data(df: pd.DataFrame, cutoff_date: dict) -> pd.DataFrame:
    """Cutoff data after a given date."""
    cutoff_abspos = compute_abspos(datetime(**cutoff_date))
    df = df[df["abspos"] <= cutoff_abspos]
    return df