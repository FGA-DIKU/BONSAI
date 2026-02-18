import torch
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from typing import List, Optional, Dict


def get_loss_weight(fn, outcomes: List[int]) -> Optional[List[float]]:
    """Get weights for weighted loss function.
    If loss_weight_function is false or undefined, then no positive weight is used.
    If loss_weight_function is defined then the function is used to calculate the weights.
    """
    if fn is None or len(outcomes) == 0:
        return None

    weight_func = instantiate(fn)
    return weight_func(outcomes)


def sqrt(outcomes: List[int]) -> float:
    """Calculate the square root of the ratio of negative to positive samples.

    Args:
        outcomes: List of binary labels (0 or 1)

    Returns:
        float: Square root of negative to positive ratio

    Raises:
        ValueError: If data contains less than 2 classes or no positive samples
    """
    labels = compute_labels(outcomes)  # Will raise ValueError if < 2 classes
    num_neg = labels.get(0, 0)
    num_pos = labels.get(1, 0)

    if num_pos == 0:
        raise ValueError("No positive samples (class 1) found in the dataset")
    if num_neg == 0:
        raise ValueError("No negative samples (class 0) found in the dataset")

    return torch.tensor(np.sqrt(num_neg / num_pos))


def effective_n_samples(outcomes: List[int]) -> float:
    """Calculate positive weight using the effective number of samples method.

    Args:
        outcomes: List of binary labels (0 or 1)

    Returns:
        float: Weight ratio based on effective number of samples

    Raises:
        ValueError: If data contains less than 2 classes or missing classes
    """
    labels = compute_labels(outcomes)  # Will raise ValueError if < 2 classes
    n0 = labels.get(0)
    n1 = labels.get(1)

    if n0 is None or n1 is None:
        raise ValueError(
            "Both classes (0 and 1) must be present for binary classification"
        )

    beta = (len(outcomes) - 1) / len(outcomes)
    alpha_0 = (1 - beta) / (1 - beta**n0)
    alpha_1 = (1 - beta) / (1 - beta**n1)
    return torch.tensor(alpha_1 / alpha_0)


def compute_labels(outcomes: List[int]) -> Dict[int, int]:
    """Compute the labels for the outcomes."""
    labels = pd.Series(outcomes)
    counts = labels.value_counts()
    print(f"Class counts:\n{counts.to_string()}")

    if len(counts) < 2:
        raise ValueError(
            f"Found only {len(counts)} class(es) in the data. Multi-class sampling/weighting requires at least 2 classes."
        )
    return counts
