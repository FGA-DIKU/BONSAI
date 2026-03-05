from collections import Counter
from typing import List, Optional, Dict
import torch
import numpy as np
from hydra.utils import instantiate


def get_loss_weight(fn, labels: List[int]) -> Optional[List[float]]:
    """Get weights for weighted loss function.
    If loss_weight_function is false or undefined, then no positive weight is used.
    If loss_weight_function is defined then the function is used to calculate the weights.
    """
    if fn is None:
        return None
    label_counts = Counter(labels)
    return instantiate(fn, label_counts=label_counts)


def sqrt(label_counts: Dict[int, int]) -> float:
    """Calculate the square root of the ratio of negative to positive samples."""
    n0 = label_counts.get(0)
    n1 = label_counts.get(1)

    if n0 is None or n1 is None:
        raise ValueError(
            f"Both classes (0: {n0} and 1: {n1}) must be present for binary classification"
        )

    return torch.tensor(np.sqrt(n0 / n1))


def effective_n_samples(label_counts: Dict[int, int]) -> float:
    """Calculate positive weight using the effective number of samples method."""
    n0 = label_counts.get(0)
    n1 = label_counts.get(1)

    if n0 is None or n1 is None:
        raise ValueError(
            f"Both classes (0: {n0} and 1: {n1}) must be present for binary classification"
        )
    n = sum(label_counts.values())
    beta = (n - 1) / n
    alpha_0 = (1 - beta) / (1 - beta**n0)
    alpha_1 = (1 - beta) / (1 - beta**n1)
    return torch.tensor(alpha_1 / alpha_0)
