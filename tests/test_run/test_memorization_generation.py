"""Public-seed sampling appends its draws and never receives patient features."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from bonsai.functional.memorization_generation import sample_codes
from bonsai.run import memorization_check as check


class TestGeneration(unittest.TestCase):
    def test_stochastic_sampling_reuses_its_own_draws_and_no_patient_features(self):
        network = nn.Linear(1, 1)
        network.pretrain_head = nn.Identity()
        network.hparams = {"max_seqlen": 16}
        seen = []

        def hidden(_model, batch, code_only):
            self.assertTrue(code_only)
            self.assertEqual(set(batch), {"code", "attention_mask"})
            seen.append(batch["code"].clone())
            return torch.zeros((*batch["code"].shape, 4))

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(check, "hidden_states", side_effect=hidden),
        ):
            first = sample_codes(
                network, 3, 32, 4, 32, 1.0, 7, Path(temporary) / "first.npy"
            )
            self.assertTrue(np.array_equal(seen[0].numpy(), np.full((32, 1), 3)))
            for position in range(1, 4):
                np.testing.assert_array_equal(
                    seen[position].numpy()[:, 1:], first[:, :position]
                )
            second = sample_codes(
                network, 3, 32, 4, 32, 1.0, 7, Path(temporary) / "second.npy"
            )
            np.testing.assert_array_equal(first, second)
            self.assertGreater(len(np.unique(first, axis=0)), 1)
