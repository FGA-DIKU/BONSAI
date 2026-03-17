import logging
from pathlib import Path
from typing import Optional
import polars as pl
from bonsai.functional.features import create_features


def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    pre = len(df)
    df = df.unique(subset=["subject_id", "code", "time"], maintain_order=True)
    if pre != len(df):
        logging.info(
            f"Dropped {pre - len(df)} duplicate rows based on subject_id, code, and time"
        )
    return df


def process_split(
    split,
    path_input_dir: Path,
    path_output_dir: Path,
    tokenizer,
    exclude_regex: Optional[str] = None,
):
    path_output_dir_split = path_output_dir / split
    path_output_dir_split.mkdir(parents=True, exist_ok=True)

    data_counts = {
        "loaded": 0,
        "after_duplicates": 0,
        "after_exclusion": 0,
        "after_features": 0,
    }
    ids = []
    shards = [shard for shard in (path_input_dir / split).glob("*.parquet")]
    logging.info(f"Found {len(shards)} shards to process in {split}")
    for shard_idx, shard in enumerate(shards, 1):
        logging.info(f"Processing shard {shard_idx}/{len(shards)}: {shard}")

        # Load
        shard_df = pl.read_parquet(shard)
        shard_df = shard_df.drop(
            "__index_level_0__"
        )  # TODO: This is probably a pandas artifact
        data_counts["loaded"] += len(shard_df)

        # Drop duplicates
        shard_df = drop_duplicates(shard_df)
        data_counts["after_duplicates"] += len(shard_df)

        # Optional: Exclude codes based on regex
        if exclude_regex is not None:
            shard_df = shard_df.filter(~pl.col("code").str.contains(exclude_regex))
        data_counts["after_exclusion"] += len(shard_df)

        # Create features
        features = create_features(shard_df)
        data_counts["after_features"] += len(features)

        # Tokenize
        tokenized = tokenizer(features)

        # Cast to correct dtypes
        tokenized = tokenized.with_columns(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("code").cast(pl.Int64),
            pl.col("age").cast(pl.Float32),
            pl.col("abspos").cast(pl.Float64),
            pl.col("segment").cast(pl.Int32),
        )
        tokenized.write_parquet(path_output_dir_split / f"{shard.stem}.parquet")

        ids.extend(tokenized["subject_id"].unique())

    logging.info(
        f"Finished processing {split} \n"
        f"Total rows loaded: {data_counts['loaded']} \n"
        f"Total rows after dropping duplicates: {data_counts['after_duplicates']} \n"
        f"Total rows after exclusion: {data_counts['after_exclusion']} \n"
        f"Total rows after feature creation: {data_counts['after_features']}"
    )

    return ids
