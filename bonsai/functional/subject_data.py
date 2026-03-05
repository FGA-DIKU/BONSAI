import logging
from pathlib import Path
from typing import List, Dict, Iterable
import torch
import pandas as pd


def prepare_subject_data(split_path: Path) -> List[Dict[str, torch.Tensor]]:
    """Load tokenized data and prepare it in the format needed for the models (subject_data)."""
    all_tokenized = []
    for shard in split_path.glob("*.parquet"):
        tokenized_data = pd.read_parquet(shard)

        # Convert to training format
        for subject_id, group in tokenized_data.groupby("subject_id", sort=False):
            all_tokenized.append(
                {
                    "subject_id": torch.tensor(subject_id),
                    "code": torch.tensor(group["code"].tolist()),
                    "abspos": torch.tensor(group["abspos"].tolist()),
                    "segment": torch.tensor(group["segment"].tolist()),
                    "age": torch.tensor(group["age"].tolist()),
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
