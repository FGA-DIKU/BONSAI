import unittest
from bonsai.functional.sampling import inverse_sqrt, effective_n_samples


class TestSampling(unittest.TestCase):
    def test_inverse_sqrt(self):
        labels = [0, 0, 1, 1, 1]
        label_counts = {0: 2, 1: 3}
        weights = inverse_sqrt(labels, label_counts)
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights[0], 1 / (2**0.5))
        self.assertAlmostEqual(weights[2], 1 / (3**0.5))

    def test_effective_n_samples(self):
        labels = [0, 0, 1, 1, 1]
        label_counts = {0: 2, 1: 3}
        weights = effective_n_samples(labels, label_counts)
        self.assertEqual(len(weights), 5)
