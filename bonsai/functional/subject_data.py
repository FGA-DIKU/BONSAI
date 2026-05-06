import logging
from pathlib import Path
from typing import List, Dict, Iterable
import torch
import polars as pl


def prepare_subject_data(split_path: Path) -> List[Dict[str, torch.Tensor]]:
    """Load tokenized data and prepare it in the format needed for the models (subject_data)."""
    all_tokenized = []
    for shard in split_path.glob("*.parquet"):
        tokenized_data = pl.read_parquet(shard)

        # Convert to training format
        for subject_id, group in tokenized_data.group_by("subject_id"):
            all_tokenized.append(
                {
                    "subject_id": subject_id[0],
                    "code": group["code"].to_torch(),
                    "abspos": group["abspos"].to_torch(),
                    "segment": group["segment"].to_torch(),
                    "age": group["age"].to_torch(),
                }
            )

    return all_tokenized


def filter_subject_data(subject_data: List[dict], cohort: Iterable[int]) -> List[dict]:
    """Filter subject_data to only include subjects in the cohort."""
    pre = len(subject_data)
    cohort_set = set(cohort)  # Convert to set for faster lookup
    filtered_subjects = [s for s in subject_data if s["subject_id"] in cohort_set]
    post = len(filtered_subjects)
    logging.info(f"filter_subject_data: {pre} -> {post} subjects")
    return filtered_subjects
