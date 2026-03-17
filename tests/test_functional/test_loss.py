import unittest
import torch
from bonsai.functional.loss import get_loss_weight, sqrt, effective_n_samples


class TestLoss(unittest.TestCase):
    def test_sqrt(self):
        label_counts = {0: 100, 1: 25}
        result = sqrt(label_counts)
        self.assertAlmostEqual(result.item(), 2.0)

    def test_effective_n_samples(self):
        label_counts = {0: 100, 1: 25}
        result = effective_n_samples(label_counts)
        self.assertIsInstance(result, torch.Tensor)

    def test_get_loss_weight_none(self):
        labels = [0, 1, 0, 1]
        result = get_loss_weight(None, labels)
        self.assertIsNone(result)
