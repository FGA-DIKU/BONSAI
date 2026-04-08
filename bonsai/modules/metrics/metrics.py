from typing import List, Literal
import torch
from torchmetrics import Metric
from bonsai.functional.metrics import fused_precision_at_k


class PrecisionAtKs(Metric):
    def __init__(
        self,
        ks: List[int],
        reduce: Literal["mean", "sum"] = "mean",
        ignore_index: int = -100,
        prefix: str = "",
        **kwargs,
    ):
        """
        Custom torchmetrics Metric to compute precision at multiple k values.

        Args:
            ks: A list of integers representing the number of top predictions to consider for computing precision.
            reduce: Method to reduce the batch precision scores. Options are "mean" or "sum".
            ignore_index: Label index to ignore when computing precision (e.g., for padding tokens).
            prefix: A string prefix to add to the metric names when logging.
        """
        super().__init__(**kwargs)
        self.ks = ks
        self.reduce = reduce
        self.ignore_index = ignore_index
        self.prefix = prefix
        self.add_state(
            "correct_counts", default=torch.zeros(len(ks)), dist_reduce_fx="sum"
        )
        self.add_state("total_counts", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        # Filter out ignored indices
        mask = labels != self.ignore_index
        logits = logits[mask]  # (B, C)
        labels = labels[mask]  # (B,)

        for idx, precision_score in enumerate(
            fused_precision_at_k(logits, labels, self.ks, reduce="sum")
        ):
            self.correct_counts[idx] += precision_score
        self.total_counts += logits.size(0)

    def compute(self):
        if self.reduce == "mean":
            return {
                f"{self.prefix}/precision@k{k}": self.correct_counts[idx]
                / self.total_counts
                for idx, k in enumerate(self.ks)
            }
        elif self.reduce == "sum":
            return {
                f"{self.prefix}/precision@k{k}": self.correct_counts[idx]
                for idx, k in enumerate(self.ks)
            }
        else:
            raise ValueError(
                f"Invalid reduce option: {self.reduce}. Must be 'mean' or 'sum'."
            )
