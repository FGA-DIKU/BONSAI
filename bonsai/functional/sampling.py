from typing import List
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import numpy as np
from hydra.utils import instantiate


def get_sampler(weight_fn, labels, label_counts):
    label_weight = instantiate(
        weight_fn,
        labels=labels,
        label_counts=label_counts,
    )
    return WeightedRandomSampler(
        weights=label_weight, num_samples=len(labels), replacement=True
    )


def inverse_sqrt(labels: list, label_counts: pd.Series) -> List[float]:
    """Calculate the inverse square root of class frequencies."""
    weights = 1 / np.sqrt(label_counts)
    # Map weights back to samples
    return [weights[label] for label in labels]


def effective_n_samples(labels: List[int], label_counts: pd.Series) -> List[float]:
    """Calculate weights using the effective number of samples method."""
    # Calculate beta as per the paper
    beta = (len(labels) - 1) / len(labels)

    # Calculate effective number for each class
    effective_nums = {
        label: (1 - (beta**count)) / (1 - beta) for label, count in label_counts.items()
    }

    # Calculate class probabilities
    total_effective = sum(effective_nums.values())
    class_probs = {
        label: eff_num / total_effective for label, eff_num in effective_nums.items()
    }

    # Calculate weights for each sample
    return [class_probs[outcome] / label_counts[outcome] for outcome in labels]
