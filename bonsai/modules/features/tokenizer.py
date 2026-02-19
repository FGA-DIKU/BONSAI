import pandas as pd
from typing import Optional, Dict

from bonsai.functional.tokenize import (
    add_special_tokens_partition,
    limit_concept_length_partition,
    tokenize_partition,
)


class EHRTokenizer:
    def __init__(
        self,
        vocabulary=None,
        cutoffs: Optional[Dict[str, int]]=None,
        sep_tokens: bool = True,
        cls_token: bool = True,
        # code_mapping: dict = None, # TODO: Dropped this
    ):
        self.new_vocab = False
        if vocabulary is None:
            self.new_vocab = True
            vocabulary = {
                "PAD_TOKEN": 0,
                "CLS_TOKEN": 1,
                "SEP_TOKEN": 2,
                "UNKNOWN_TOKEN": 3,
                "MASK_TOKEN": 4,
        }
        self.vocabulary = vocabulary

        self.cutoffs = cutoffs
        self.sep_tokens = sep_tokens
        self.cls_token = cls_token
        # self.code_mapping = code_mapping

    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        !We assume that features are sorted by PID and abspos and PID is the index.
        """
        # # Apply code mapping if provided
        # if self.code_mapping is not None:
        #     features["code"] = features["code"].map(
        #         lambda x: self.code_mapping.get(x, x)
        #     )

        # Apply cutoffs if needed before updating vocabulary
        if self.cutoffs is not None:
            features["code"] = limit_concept_length_partition(
                features["code"], self.cutoffs
            )

        # Update vocabulary with concepts after cutoffs
        if self.new_vocab:
            self.update_vocabulary(features["code"])

        # Combine all operations into a single partition pass
        def _process_partition(df):
            # Add special tokens
            if self.sep_tokens or self.cls_token:
                df = add_special_tokens_partition(
                    df, add_sep=self.sep_tokens, add_cls=self.cls_token
                )
            # Tokenize within the same partition
            df["code"] = tokenize_partition(df["code"], self.vocabulary)
            return df

        # return features.map_partitions(_process_partition)
        return _process_partition(features)

    def update_vocabulary(self, concepts: pd.Series) -> None:
        """Create or update vocabulary from unique concepts"""
        # Get unique concepts across all partitions
        unique_concepts = concepts.drop_duplicates()

        # Start with base vocabulary
        vocabulary = self.vocabulary.copy()
        # Add new concepts
        new_concepts = set(unique_concepts) - set(vocabulary)
        # Pre-calculate the starting index
        start_idx = max(vocabulary.values()) + 1
        if new_concepts:
            new_indices = range(start_idx, start_idx + len(new_concepts))
            vocabulary.update(dict(zip(new_concepts, new_indices)))
        self.vocabulary = vocabulary

    def freeze_vocabulary(self) -> None:
        self.new_vocab = False
