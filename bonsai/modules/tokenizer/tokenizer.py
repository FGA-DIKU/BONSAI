import pandas as pd
from typing import Optional, Dict


class EHRTokenizer:
    def __init__(
        self,
        vocabulary=None,
        cutoffs: Optional[Dict[str, int]]=None,
        sep_tokens: bool = True,
    ):
        self.hot_vocab = vocabulary is None
        if vocabulary is None:
            vocabulary = {
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[UNK]": 3,
                "[MASK]": 4,
        }
        self.vocabulary = vocabulary

        self.cutoffs = cutoffs
        self.sep_tokens = sep_tokens

    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        !We assume that features are sorted by subject_id and abspos.
        """
        # Apply cutoffs if needed before updating vocabulary
        if self.cutoffs is not None:
            features["code"] = self.limit_code_length(features["code"])

        # Update vocabulary if vocabulary is `hot`
        if self.hot_vocab:
            self.update_vocabulary(features["code"])

        if self.sep_tokens:
            features = self.add_sep_tokens(features)

        # Tokenize
        features["code"] = self.tokenize(features["code"])

        return features

    def update_vocabulary(self, codes: pd.Series) -> None:
        """Update self.vocabulary from unique codes"""
        # Get unique codes
        unique_concepts = codes.unique()

        # Add new concepts
        new_concepts = set(unique_concepts) - set(self.vocabulary)
        if new_concepts:
            start_idx = max(self.vocabulary.values()) + 1
            new_indices = range(start_idx, start_idx + len(new_concepts))
            self.vocabulary.update(dict(zip(new_concepts, new_indices)))

    def add_sep_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add [SEP] tokens at segment changes within the same subject_id"""
        pid_series = df["subject_id"]
        segment_changes = (df["segment"] != df["segment"].shift(-1)) & (
            pid_series == pid_series.shift(-1)
        )
        sep_rows = df[segment_changes].copy()
        sep_rows["code"] = "[SEP]"
        df = pd.concat([df, sep_rows], ignore_index=True)
        df = df.sort_values(["subject_id", "abspos"]).reset_index(drop=True)
        return df
    
    def tokenize(self, codes: pd.Series) -> pd.Series:
        """Tokenizes a series using self.vocabulary, mapping unknown codes to [UNK] token"""
        return codes.map(self.vocabulary).fillna(self.vocabulary["[UNK]"]).astype(int)
    
    def limit_code_length(self, codes: pd.Series) -> pd.Series:
        """Limit concept lengths using a {prefix: length} self.cutoff dict.
        Example:
            With cutoffs={'D': 4}, 'D123456' becomes 'D1234'
        """
        for prefix, length in self.cutoffs.items():
            # Create mask for matching prefix
            mask = codes.str.startswith(prefix)
            codes[mask] = codes[mask].str[:length]

        return codes

    def freeze_vocabulary(self) -> None:
        self.hot_vocab = False

    def hot_vocabulary(self) -> None:
        self.hot_vocab = True
