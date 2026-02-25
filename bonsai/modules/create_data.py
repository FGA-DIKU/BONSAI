import pandas as pd
from pathlib import Path
from typing import Optional
import pyarrow as pa

from bonsai.functional.features import create_features
from bonsai.functional.io_ops import load_concept
from bonsai.functional.regex_utils import exclude_codes
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer


def create_features_and_tokenize(
    split: str,
    path_data: Path,
    path_tokenized: Path,
    logger,
    tokenizer: EHRTokenizer,
    exclude_regex: Optional[str] = None,
) -> None:
    """
    Creates features and tokenizes them, saving the tokenized results to disk.
    """
    logger.info(f"create_features_and_tokenize: {split}")

    path_tokenized_split = path_tokenized / split
    path_tokenized_split.mkdir(parents=True, exist_ok=True)

    shards = [shard for shard in (path_data / split).iterdir()]
    logger.info(f"Found {len(shards)} shards to process in {split}")

    concept_counts = {
        "loaded": 0,
        "after_duplicates": 0,
        "after_exclusion": 0,
        "after_features": 0,
    }
    for shard_idx, shard in enumerate(shards, 1):
        logger.info(f"Processing shard {shard_idx}/{len(shards)}: {shard}")

        concepts = load_concept(shard)
        concept_counts["loaded"] += len(concepts)

        concepts = drop_duplicates(concepts, logger)
        concept_counts["after_duplicates"] += len(concepts)

        if exclude_regex is not None:
            concepts = exclude_codes(concepts, exclude_regex)
        concept_counts["after_exclusion"] += len(concepts)

        features = create_features(concepts, logger)
        concept_counts["after_features"] += len(features)

        tokenized = tokenizer(features)

        tokenized.to_parquet(
            path_tokenized_split / f"{shard.stem}.parquet",
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
    logger.info(f"Finished processing {split}")
    logger.info(f"Total concepts loaded: {concept_counts['loaded']}")
    logger.info(
        f"Total concepts after dropping duplicates: {concept_counts['after_duplicates']}"
    )
    logger.info(f"Total concepts after exclusion: {concept_counts['after_exclusion']}")
    logger.info(
        f"Total concepts after feature creation: {concept_counts['after_features']}"
    )


def drop_duplicates(concepts: pd.DataFrame, logger) -> pd.DataFrame:
    pre = len(concepts)
    concepts = concepts.drop_duplicates(subset=["subject_id", "code", "time"])
    if pre != len(concepts):
        logger.info(
            f"Dropped {pre - len(concepts)} duplicate rows based on subject_id, code, and time"
        )
    return concepts
