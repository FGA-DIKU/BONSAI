import pandas as pd


def add_special_tokens_partition(
    df: pd.DataFrame, add_sep=True, add_cls=True
) -> pd.DataFrame:
    """
    Efficiently add special tokens to a partition without full sorting.
    PID is assumed to be the index.

    cls token will be added before earliest abspos for each PID
    sep token will be added at segment changes, adjacent to the last event of the previous segment
    """
    special_rows = []

    if add_cls:
        # Find indices of the earliest event for each PID
        cls_rows = df.groupby("subject_id").first()
        # Create [CLS] rows
        cls_rows["code"] = "[CLS]"
        cls_rows["abspos"] -= (
            1e-3  # Adjust position to come before earliest event
        )
        cls_rows["segment"] = 0
        special_rows.append(cls_rows)

    if add_sep:
        # Find segment changes within same PID
        df = df.sort_values(["subject_id", "abspos", "segment"])
        pid_series = df.index.to_series()
        segment_changes = (df["segment"] != df["segment"].shift(-1)) & (
            pid_series == pid_series.shift(-1)
        )
        sep_rows = df[segment_changes].copy()
        sep_rows["code"] = "[SEP]"
        sep_rows["abspos"] += (
            1e-3   # Adjust position slightly
        )
        special_rows.append(sep_rows)

    # Combine all rows and sort by 'PID' and 'abspos'
    if special_rows:
        df = pd.concat([df] + special_rows, ignore_index=False)
        df = df.sort_values(["subject_id", "abspos"])

    return df


def tokenize_partition(series: pd.Series, vocabulary: dict) -> pd.Series:
    """Optimized in-partition tokenization using direct dictionary mapping."""
    # Direct mapping with fillna for unknown tokens
    return series.map(vocabulary).fillna(vocabulary["[UNK]"]).astype(int)


def limit_concept_length_partition(series: pd.Series, cutoffs: dict) -> pd.Series:
    """Efficiently limit concept lengths within a partition.

    Args:
        series: Pandas Series containing concepts
        cutoffs: Dict mapping prefixes to max lengths, e.g. {'D': 6, 'M': 4}
            Will limit concepts starting with 'D' to 6 chars, 'M' to 4 chars.

    Example:
        With cutoffs={'D': 4}, 'D123456' becomes 'D1234'
    """
    # Create a copy to avoid modifying original
    result = series.copy()

    # Vectorized operations for each prefix
    for prefix, length in cutoffs.items():
        # Create mask for matching prefix
        mask = result.str.startswith(prefix)

        # Apply length limit only where mask is True
        if mask.any():
            result.loc[mask] = result.loc[mask].str[:length]

    return result
