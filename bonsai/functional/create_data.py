import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
from bonsai.functional.features import create_features
from bonsai.functional.regex_utils import exclude_codes


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    pre = len(df)
    df = df.drop_duplicates(subset=["subject_id", "code", "time"])
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
        shard_df = pd.read_parquet(shard)
        data_counts["loaded"] += len(shard_df)

        # Drop duplicates
        shard_df = drop_duplicates(shard_df)
        data_counts["after_duplicates"] += len(shard_df)

        # Optional: Exclude codes based on regex
        if exclude_regex is not None:
            shard_df = exclude_codes(shard_df, exclude_regex)
        data_counts["after_exclusion"] += len(shard_df)

        # Create features
        features = create_features(shard_df)
        data_counts["after_features"] += len(features)

        # Tokenize
        tokenized = tokenizer(features)

        tokenized.to_parquet(
            path_output_dir_split / f"{shard.stem}.parquet",
            index=False,
            schema=pa.schema(
                {
                    "subject_id": "int64",
                    "code": "int64",
                    "age": "float32",
                    "abspos": "float64",
                    "segment": "int32",
                }
            ),
        )

        ids.extend(tokenized["subject_id"].unique())

    logging.info(
        f"Finished processing {split} \n"
        f"Total rows loaded: {data_counts['loaded']} \n"
        f"Total rows after dropping duplicates: {data_counts['after_duplicates']} \n"
        f"Total rows after exclusion: {data_counts['after_exclusion']} \n"
        f"Total rows after feature creation: {data_counts['after_features']}"
    )

    return ids
