import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict

def prepare_subject_data(split_path: Path, logger) -> List[Dict[str, torch.Tensor]]:
    """Load tokenized data and prepare it in the format needed for the models (subject_data)."""
    all_tokenized = []
    for shard in split_path.glob("*.parquet"):
        logger.info(f"Preparing training format: {split_path.name}/{shard.name}")
        tokenized_data = pd.read_parquet(shard)

        # Convert to training format
        for subject_id, group in tokenized_data.groupby("subject_id", sort=False):
            all_tokenized.append({
                "subject_id": subject_id,
                "codes": torch.tensor(group["code"].tolist()),
                "abspos": torch.tensor(group["abspos"].tolist()),
                "segments": torch.tensor(group["segment"].tolist()),
                "ages": torch.tensor(group["age"].tolist()),
            })

    return all_tokenized


def filter_subject_data(subject_data: List[dict], cohort: list, logger) -> List[dict]:
    """Filter subject_data to only include subjects in the cohort."""
    pre = len(subject_data)
    cohort = set(cohort)  # Convert to set for faster lookup
    filtered_subjects = [s for s in subject_data if s["subject_id"] in cohort]
    post = len(filtered_subjects)
    logger.info(f"Cohort filtering: {pre} -> {post} subjects")
    return filtered_subjects
