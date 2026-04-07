from typing import Literal, List


def precision_at_k(logits, labels, k: int, reduce: Literal["mean", "sum"] = "mean"):
    """
    Computes precision at k for a batch of predictions and labels.

    Args:
        logits: Tensor of shape (B, C) containing predicted logits for each class.
        labels: Tensor of shape (B,) containing true class labels.
        k: The number of top predictions to consider for computing precision.
        reduce: Method to reduce the batch precision scores. Options are "mean" or "sum".

    Returns:
        Precision at k score as a float.
    """
    top_k_preds = logits.topk(k, dim=-1).indices  # (B, k)
    correct = top_k_preds.eq(labels.unsqueeze(-1))  # (B, k)
    precision_scores = correct.any(dim=-1).float()  # (B,)

    if reduce == "mean":
        return precision_scores.mean().item()
    elif reduce == "sum":
        return precision_scores.sum().item()
    else:
        raise ValueError(f"Invalid reduce option: {reduce}. Must be 'mean' or 'sum'.")


def fused_precision_at_k(
    logits, labels, k: List[int], reduce: Literal["mean", "sum"] = "mean"
):
    """
    Computes precision for multiple k values for a batch of logits and labels using a fused implementation.
    By only computing the top max(k) predictions once, we can efficiently compute precision for multiple k values.

    Args:
        logits: Tensor of shape (B, C) containing predicted logits for each class.
        labels: Tensor of shape (B,) containing true class labels.
        k: A list of integers representing the number of top predictions to consider for computing precision.
        reduce: Method to reduce the batch precision scores. Options are "mean" or "sum".

    Returns:
        Yields tuples of (k, precision_score) for each k value.
    """
    max_k = max(k)
    top_k_preds = logits.topk(max_k, dim=-1).indices  # (B, max_k)
    for k_val in k:
        correct = top_k_preds[:, :k_val].eq(labels.unsqueeze(-1))  # (B, k_val)
        precision_scores = correct.any(dim=-1).float()  # (B,)
        if reduce == "mean":
            yield k_val, precision_scores.mean().item()
        elif reduce == "sum":
            yield k_val, precision_scores.sum().item()
        else:
            raise ValueError(
                f"Invalid reduce option: {reduce}. Must be 'mean' or 'sum'."
            )
