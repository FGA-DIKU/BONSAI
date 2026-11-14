"""Every-step metrics agree with the existing prefix scores and intervals."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from bonsai.functional.memorization_metrics import difference_in_means, score_patient
from bonsai.run.zero_context_falloff import (
    COHORTS,
    METRICS,
    VIEWS,
    Moments,
    plot_curves,
    prefix_scores,
    summarize,
    write_report,
)


class TestFalloff(unittest.TestCase):
    def test_every_prefix_matches_existing_metrics_with_repetition_and_separators(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            truth, predicted = rng.integers(0, 8, size=(2, 37))
            values = prefix_scores(predicted, truth, {1, 2, 3, 4, 5}, {1, 3})
            for step in range(len(truth)):
                expected = score_patient(
                    predicted[: step + 1], truth[: step + 1], {1, 2, 3, 4, 5}, {1, 3}
                )
                np.testing.assert_allclose(
                    values[step, :3], [expected[k] for k in METRICS[:3]], equal_nan=True
                )
                self.assertEqual(values[step, 3], predicted[step] == truth[step])
            self.assertTrue((np.diff(values[:, 0]) <= 0).all())

    def test_online_summaries_match_patient_arrays_and_preserve_missing_rare_support(
        self,
    ):
        rng = np.random.default_rng(4)
        groups = {
            (v, c, s): Moments(7)
            for v in VIEWS
            for c in COHORTS
            for s in ("train", "tuning")
        }
        patients = {}
        for split in ("train", "tuning"):
            patients[split] = []
            for i in range(20):
                truth, predicted = rng.integers(0, 5, size=(2, 7))
                values = prefix_scores(predicted, truth, {1, 2, 3, 4}, set())
                patients[split].append(values)
                for view in VIEWS:
                    groups[view, "combined", split].update(values)
        frame = summarize(groups, ("train", "tuning"), 7, 2)
        for row in frame.filter(
            (pl.col("view") == "fixed") & (pl.col("cohort") == "combined")
        ).iter_rows(named=True):
            step, metric = row["Y"] - 1, METRICS.index(row["score"])
            if row["score"] == "rare_codes":
                self.assertEqual(row["patients_train"], 0)
                self.assertTrue(np.isnan(row["lower"]))
                continue
            arrays = [
                np.asarray(patients[s])[:, step, metric] for s in ("train", "tuning")
            ]
            arrays = [values[np.isfinite(values)] for values in arrays]
            self.assertEqual(row["patients_train"], len(arrays[0]))
            self.assertEqual(row["patients_unseen"], len(arrays[1]))
            expected = difference_in_means(
                *arrays, row["score"] in ("exact_match", "position_accuracy")
            )
            np.testing.assert_allclose(
                [row[k] for k in ("excess", "lower", "upper")], expected, atol=1e-14
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_report(frame, root, 7)
            self.assertIn(
                "Original check statuses are unchanged",
                (root / "report.txt").read_text(),
            )
            if importlib.util.find_spec("matplotlib"):
                plot_curves(frame, root)
                self.assertEqual(len(list(root.glob("*.png"))), 3)
