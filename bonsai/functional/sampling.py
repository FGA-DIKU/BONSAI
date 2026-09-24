from collections import Counter
from typing import List

import numpy as np
from hydra.utils import instantiate
from torch.utils.data import WeightedRandomSampler


def get_sampler(weight_fn, labels) -> WeightedRandomSampler:
    if weight_fn is None:
        return None
    label_counts = Counter(labels)
    label_weight = instantiate(
        weight_fn,
        labels=labels,
        label_counts=label_counts,
    )
    return WeightedRandomSampler(
        weights=label_weight, num_samples=len(labels), replacement=True
    )


def inverse_sqrt(labels: List[int], label_counts: dict) -> List[float]:
    """Calculate the inverse square root of class frequencies."""
    weights = {k: 1 / np.sqrt(v) for k, v in label_counts.items()}
    # Map weights back to samples
    return [weights[label] for label in labels]


def effective_n_samples(labels: List[int], label_counts: dict) -> List[float]:
    """Calculate weights using the effective number of samples method."""
    # Calculate beta as per the paper
    beta = (len(labels) - 1) / len(labels)

    # Calculate effective number for each class
    effective_nums = {
        label: (1 - (beta**count)) / (1 - beta) for label, count in label_counts.items()
    }

    return [1.0 / effective_nums[label] for label in labels]

