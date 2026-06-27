from typing import Literal
import torch
from torchmetrics import Metric
from torchmetrics.classification import BinarySensitivityAtSpecificity


class BinarySensAtSpec(Metric):
    """Sensitivity at minimum specificity; returns a single tensor for Lightning logging."""

    full_state_update: bool = False

    def __init__(self, min_specificity: float = 0.85, **kwargs):
        super().__init__(**kwargs)
        self.metric = BinarySensitivityAtSpecificity(min_specificity=min_specificity)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        probs = torch.sigmoid(preds.squeeze(-1).float())
        self.metric.update(probs, target.squeeze(-1).int())

    def compute(self) -> torch.Tensor:
        sens, _ = self.metric.compute()
        return sens

    def reset(self) -> None:
        self.metric.reset()


class SharedPrecisionAtK(Metric):
    def __init__(
        self,
        k: int,
        max_k: int,
        reduce: Literal["mean", "sum"] = "mean",
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert k <= max_k, "k must be less than or equal to max_k"
        self.k = k
        self.max_k = max_k
        self.reduce = reduce
        self.ignore_index = ignore_index
        self.add_state(
            "correct_counts",
            default=torch.zeros(max_k, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_count",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        # Filter out ignored indices
        mask = labels != self.ignore_index
        logits = logits[mask]  # (B, C)
        labels = labels[mask]  # (B,)

        top_k_preds = logits.topk(self.max_k, dim=-1).indices  # (B, max_k)
        correct = top_k_preds.eq(labels.unsqueeze(-1))  # (B, max_k)
        self.correct_counts += correct.sum(dim=0).cumsum(
            dim=0
        )  # Cumulative sum to get correct counts for all k up to max_k
        self.total_count += logits.size(0)

    def compute(self):
        if self.reduce == "mean":
            return self.correct_counts[self.k - 1] / self.total_count
        elif self.reduce == "sum":
            return self.correct_counts[self.k - 1]
        else:
            raise ValueError(
                f"Invalid reduce option: {self.reduce}. Must be 'mean' or 'sum'."
            )
