import unittest
import torch
from bonsai.functional.normalization import normalize_segments


class TestNormalization(unittest.TestCase):
    def test_normalize_segments(self):
        segments = torch.tensor([5, 5, 8, 8, 10])
        normalized = normalize_segments(segments)
        self.assertTrue(torch.equal(normalized, torch.tensor([0, 0, 1, 1, 2])))
