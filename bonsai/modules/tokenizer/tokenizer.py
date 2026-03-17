from typing import Optional, Dict
import polars as pl


class EHRTokenizer:
    def __init__(
        self,
        vocabulary=None,
        cutoffs: Optional[Dict[str, int]] = None,
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

    def __call__(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        !We assume that features are sorted by subject_id and abspos.
        """
        # Apply cutoffs if needed before updating vocabulary
        if self.cutoffs is not None:
            features = features.with_columns(self.limit_code_length(pl.col("code")))

        # Update vocabulary if vocabulary is `hot`
        if self.hot_vocab:
            self.update_vocabulary(features["code"])

        if self.sep_tokens:
            features = self.add_sep_tokens(features)

        # Tokenize
        features = features.with_columns(self.tokenize(pl.col("code")))

        return features

    def update_vocabulary(self, codes: pl.Series) -> None:
        """Update self.vocabulary from unique codes"""
        # Get unique codes
        unique_codes = codes.unique()

        # Add new codes
        new_codes = set(unique_codes) - set(self.vocabulary)
        if new_codes:
            start_idx = max(self.vocabulary.values()) + 1
            new_indices = range(start_idx, start_idx + len(new_codes))
            self.vocabulary.update(dict(zip(new_codes, new_indices)))

    def add_sep_tokens(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add [SEP] tokens at segment changes within the same subject_id"""
        sep_rows = df.filter(
            (pl.col("segment") != pl.col("segment").shift(-1))
            & (pl.col("subject_id") == pl.col("subject_id").shift(-1))
        ).with_columns(code=pl.lit("[SEP]"))
        df = pl.concat([df, sep_rows])
        df = df.sort(["subject_id", "abspos"])
        return df

    def tokenize(self, codes: pl.Expr) -> pl.Expr:
        """Map self.vocabulary onto codes, mapping unknown codes to [UNK] token"""
        return codes.replace_strict(self.vocabulary, default=self.vocabulary["[UNK]"])

    def limit_code_length(self, codes: pl.Expr) -> pl.Expr:
        """Limit code lengths using a {prefix: length} self.cutoff dict.
        Example:
            With cutoffs={'D': 4}, 'D123456' becomes 'D1234'
        """
        for prefix, length in self.cutoffs.items():
            codes = (
                pl.when(codes.str.starts_with(prefix))
                .then(codes.str.slice(0, length))
                .otherwise(codes)
            )

        return codes

    def freeze_vocabulary(self) -> None:
        self.hot_vocab = False

    def hot_vocabulary(self) -> None:
        self.hot_vocab = True
