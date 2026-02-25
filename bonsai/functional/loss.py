import torch
import numpy as np
from hydra.utils import instantiate
from typing import List, Optional


def get_loss_weight(fn, label_counts: List[int]) -> Optional[List[float]]:
    """Get weights for weighted loss function.
    If loss_weight_function is false or undefined, then no positive weight is used.
    If loss_weight_function is defined then the function is used to calculate the weights.
    """
    if fn is None:
        return None

    return instantiate(fn, label_counts=label_counts)


def sqrt(label_counts: List[int]) -> float:
    """Calculate the square root of the ratio of negative to positive samples."""
    num_neg = label_counts.get(0, 0)
    num_pos = label_counts.get(1, 0)

    if num_pos == 0:
        raise ValueError("No positive samples (class 1) found in the dataset")
    if num_neg == 0:
        raise ValueError("No negative samples (class 0) found in the dataset")

    return torch.tensor(np.sqrt(num_neg / num_pos))


def effective_n_samples(label_counts: List[int]) -> float:
    """Calculate positive weight using the effective number of samples method."""
    n0 = label_counts.get(0)
    n1 = label_counts.get(1)

    if n0 is None or n1 is None:
        raise ValueError(
            "Both classes (0 and 1) must be present for binary classification"
        )

    beta = (label_counts.sum() - 1) / label_counts.sum()
    alpha_0 = (1 - beta) / (1 - beta**n0)
    alpha_1 = (1 - beta) / (1 - beta**n1)
    return torch.tensor(alpha_1 / alpha_0)
