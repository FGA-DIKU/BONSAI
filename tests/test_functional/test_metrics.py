import unittest
import torch

from bonsai.functional.metrics import fused_precision_at_k, precision_at_k


class TestMetrics(unittest.TestCase):
    def test_precision_at_k_mean(self):
        logits = torch.tensor(
            [
                [10.0, 1.0, 0.0],
                [0.0, 5.0, 4.0],
                [0.0, 0.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 2, 1])

        result = precision_at_k(logits, labels, k=1, reduce="mean")
        self.assertAlmostEqual(result, 1.0 / 3.0)

        result_top2 = precision_at_k(logits, labels, k=2, reduce="mean")
        self.assertAlmostEqual(result_top2, 2.0 / 3.0)

    def test_precision_at_k_sum(self):
        logits = torch.tensor(
            [
                [0.5, 1.0, 0.0],
                [0.0, 2.0, 1.0],
            ]
        )
        labels = torch.tensor([1, 2])

        result = precision_at_k(logits, labels, k=2, reduce="sum")
        self.assertEqual(result, 2)

    def test_fused_precision_at_k_matches_precision_at_k(self):
        logits = torch.tensor(
            [
                [5.0, 4.0, 3.0],
                [0.0, 1.0, 2.0],
                [2.0, 0.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 2, 2])
        ks = [1, 2, 3]

        fused_results = list(fused_precision_at_k(logits, labels, ks, reduce="mean"))
        expected = [
            precision_at_k(logits, labels, k=k_val, reduce="mean") for k_val in ks
        ]

        self.assertEqual([k for k, _ in fused_results], ks)
        self.assertEqual([score for _, score in fused_results], expected)
