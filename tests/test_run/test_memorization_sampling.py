"""Independent rarity reference, fixed IDs, cache reuse and cohort reports."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
from test_memorization_check import model, patient, save_checkpoint

from bonsai.functional.memorization_cache import (
    FixedCohortWindows,
    read_cache,
    save_cache,
)
from bonsai.functional.memorization_sampling import (
    SamplingSettings,
    SequenceReference,
    select_cohorts,
)
from bonsai.run import memorization_check as check


def population():
    rng = np.random.default_rng(2026)
    histories = {
        split: [rng.integers(1, 7, 40) for _ in range(count)]
        for split, count in (("train", 600), ("tuning", 400))
    }
    return histories


class TestFixedSampling(unittest.TestCase):
    def test_zero_observed_variance_does_not_create_zero_width_mean_interval(self):
        for score in (0.0, 0.5, 1.0):
            mean, lower, upper = check.difference_in_means(
                np.full(1000, score), np.full(1000, score), False
            )
            self.assertEqual(mean, 0)
            self.assertLess(lower, 0)
            self.assertGreater(upper, 0)
        _, low_small, high_small = check.difference_in_means(
            np.zeros(100), np.zeros(100), False
        )
        _, low_large, high_large = check.difference_in_means(
            np.zeros(1000), np.zeros(1000), False
        )
        self.assertLess(high_large - low_large, high_small - low_small)

    def test_candidate_shortage_is_reported_without_relaxing_rarity(self):
        histories = population()
        lengths = {
            s: np.array([len(row) for row in rows]) for s, rows in histories.items()
        }
        settings = SamplingSettings(
            patients_per_split=20,
            reference_patients=60,
            calibration_patients=60,
            span=4,
            minimum_clinical_tokens=1,
            minimum_calibration=10,
            candidate_limit=1,
        )
        read = lambda split, index, stop: histories[split][index][:stop]
        selected, audit = select_cohorts(
            lengths, read, set(range(1, 7)), [2, 8], 3, settings
        )
        for split in histories:
            self.assertFalse(audit["splits"][split]["complete"])
            self.assertEqual(audit["splits"][split]["candidates_scored"], 1)
            self.assertLessEqual(
                sum(row["cohort"] == "uncommon" for row in selected[split]), 1
            )
        with self.assertRaisesRegex(ValueError, "distinct upper rarity tail"):
            select_cohorts(
                lengths,
                lambda split, index, stop: np.ones(min(stop, 40), dtype=int),
                {1},
                [2, 8],
                3,
                replace(settings, candidate_limit=400),
            )

    def test_reference_is_independent_and_repetition_is_capped(self):
        settings = SamplingSettings(span=4, minimum_clinical_tokens=1)
        reference = SequenceReference({1, 2, 3, 4}, [1, 5], settings)
        reference.fit([[1, 2, 1, 2, 1, 2, 1, 2, 1]])
        self.assertEqual(reference.pairs[1 * reference.radix + 2], 1)
        self.assertEqual(reference.pairs[2 * reference.radix + 1], 1)
        self.assertEqual(reference.observations, 2)
        common = reference.score([1, 2, 1, 2, 1])[1]
        unusual = reference.score([1, 3, 4, 3, 4])[1]
        self.assertGreater(unusual, common)

    def test_fixed_selection_disjointness_shared_quotas_and_determinism(self):
        histories = population()
        lengths = {
            s: np.array([len(row) for row in rows]) for s, rows in histories.items()
        }
        settings = SamplingSettings(
            patients_per_split=20,
            reference_patients=60,
            calibration_patients=60,
            span=4,
            minimum_clinical_tokens=1,
            minimum_calibration=10,
            candidate_limit=400,
        )
        read = lambda split, index, stop: histories[split][index][:stop]
        selected, audit = select_cohorts(
            lengths, read, set(range(1, 7)), [2, 8], 3, settings
        )
        again, _ = select_cohorts(lengths, read, set(range(1, 7)), [2, 8], 3, settings)
        self.assertEqual(selected, again)
        reserved = set(audit["reference_indices"]) | set(audit["calibration_indices"])
        self.assertFalse(
            set(audit["reference_indices"]) & set(audit["calibration_indices"])
        )
        self.assertFalse(reserved & {row["index"] for row in selected["train"]})
        for split in histories:
            self.assertEqual(len({row["index"] for row in selected[split]}), 20)
            self.assertEqual(
                sum(row["cohort"] == "random" for row in selected[split]), 16
            )
            self.assertTrue(audit["splits"][split]["complete"])
            self.assertEqual(
                audit["splits"][split]["uncommon_by_stratum"],
                {
                    key: value
                    for key, value in audit["splits"][split]["quota_by_stratum"].items()
                    if value
                },
            )

    def test_fixed_cache_filters_after_selecting_ids_and_keeps_all_loss_windows(self):
        rows = [
            dict(
                patient(i, [1] * length),
                sampling_cohort="random" if i < 8 else "uncommon",
            )
            for i, length in enumerate([3, 6, 10, 20, 30, 40, 50, 60, 8, 70])
        ]
        payload = {
            "rows": rows,
            "members": [],
            "random_fraction": 0.8,
            "excluded_target": None,
        }
        fixed = FixedCohortWindows(payload)
        small = {row["subject_id"] for row in fixed.sample(0, 5, False)}
        long = {row["subject_id"] for row in fixed.sample(20, 5, False)}
        self.assertTrue(long <= small)
        self.assertEqual(
            {row["subject_id"] for row in fixed.sample(2, 10, True)}, set(range(10))
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            save_cache(
                path,
                {"design": "fixed"},
                10,
                {"train": payload},
                {1},
                set(),
                {"reserved": 20},
            )
            cached, _, _, _ = read_cache(path, {"design": "fixed"}, 10)
            self.assertIsInstance(cached["train"], FixedCohortWindows)
            self.assertEqual(cached["train"].sampling_audit, {"reserved": 20})
            self.assertEqual(
                [len(row["code"]) for row in cached["train"].sample(2, 10, True)],
                [len(row["code"]) for row in rows],
            )

    def test_interval_gate_flags_both_directions_but_not_shared_reproduction(self):
        scores = pl.DataFrame(
            [
                {
                    "mode": mode,
                    "split": split,
                    "X": 32,
                    "Y": 32,
                    "exact_match": 1.0,
                    "overlap": 1.0,
                    "rare_codes": 1.0,
                }
                for mode in check.MODES
                for split in ("train", "tuning")
                for _ in range(20)
            ]
        )
        coverage = check.grid_coverage(scores, [32], [32], ("train", "tuning"), 128, 5)
        loss = check.membership_from_loss(np.ones(20), np.ones(20))
        excess = check.excess_over_unseen(scores, "train", "tuning")
        self.assertTrue(
            check.comparison_summary(
                excess, loss, 5, check.patient_tails(scores), coverage
            ).startswith("CHECKS PASSED:")
        )
        negative = excess.with_columns(
            pl.lit(-0.2).alias("lower"), pl.lit(-0.1).alias("upper")
        )
        self.assertTrue(
            check.comparison_summary(negative, loss, 5, None, coverage).startswith(
                "CHECK:"
            )
        )
        self.assertTrue(
            check.comparison_summary(
                excess.head(0), loss, 5, None, coverage
            ).startswith("INCONCLUSIVE:")
        )
        filtered_coverage = check.grid_coverage(
            scores, [32], [1, 2, 32], ("train", "tuning"), 128, 5, minimum_diversity=3
        )
        self.assertEqual(
            filtered_coverage.filter(
                pl.col("status") == "not_applicable_diversity"
            ).height,
            12,
        )
        self.assertTrue(
            check.comparison_summary(
                excess, loss, 5, None, filtered_coverage
            ).startswith("CHECKS PASSED:")
        )
        no_applicable = check.grid_coverage(
            scores.head(0),
            [32],
            [1, 2],
            ("train", "tuning"),
            128,
            5,
            minimum_diversity=3,
        )
        self.assertTrue(
            check.comparison_summary(
                excess.head(0), loss, 5, None, no_applicable
            ).startswith("INCONCLUSIVE:")
        )

    def test_real_cli_sampling_preparation_cache_and_three_cohort_reports(self):
        histories = population()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(root / "model.ckpt", model())
            torch.save(
                {"[PAD]": 0, "[SEP]": 7, **{f"C{i}": i for i in range(1, 7)}},
                root / "vocabulary.pt",
            )
            for split, rows in histories.items():
                offset = 0 if split == "train" else 10000
                torch.save(
                    [patient(offset + i, row.tolist()) for i, row in enumerate(rows)],
                    root / f"subject_data_{split}.pt",
                )
            (root / "config.yaml").write_text(
                "paths:\n  population: null\ntraining:\n  cutoff_date: null\n  max_len: 32\n"
            )
            command = [
                sys.executable,
                "-m",
                "bonsai.run.memorization_check",
                "--checkpoint",
                str(root / "model.ckpt"),
                "--data-dir",
                str(root),
                "--run-config",
                str(root / "config.yaml"),
                "--device",
                "cpu",
                "--sampling-design",
                "fixed",
                "--prompt-events",
                "2",
                "8",
                "--continuation-events",
                "1",
                "4",
                "--max-patients",
                "20",
                "--cache-patients",
                "20",
                "--min-patients",
                "2",
                "--rarity-reference-patients",
                "60",
                "--rarity-calibration-patients",
                "60",
                "--rarity-span",
                "4",
                "--rarity-min-clinical-tokens",
                "1",
                "--rarity-min-calibration",
                "10",
                "--rarity-candidate-limit",
                "400",
                "--evaluation-cache",
                str(root / "cache.pt"),
                "--sep-excluded-from-training-loss",
                "--export-patient-inspection",
            ]
            preparation = subprocess.run(
                [
                    *command,
                    "--prepare-sampling-only",
                    "--output-dir",
                    str(root / "prepare"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(preparation.returncode, 0, preparation.stderr)
            self.assertNotIn("up to Y=", preparation.stdout)
            self.assertFalse((root / "prepare/excess.csv").exists())
            self.assertEqual(
                pl.read_csv(root / "prepare/sampling_patients.csv").height, 40
            )
            completed = subprocess.run(
                [*command, "--output-dir", str(root / "full")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("Loading train metadata", completed.stdout)
            self.assertNotIn("Fitting rarity reference", completed.stdout)
            manifest = json.loads((root / "full/run_manifest.json").read_text())
            self.assertTrue(manifest["evaluation_cache"]["reused"])
            self.assertEqual(manifest["sampling_design"], "fixed")
            audit = torch.load(root / "cache.pt", weights_only=False, mmap=True)[
                "sampling"
            ]
            train_indices = set(
                pl.read_csv(root / "full/sampling_patients.csv").filter(
                    pl.col("split") == "train"
                )["subject_id"]
            )
            self.assertFalse(train_indices & set(audit["reference_indices"]))
            self.assertFalse(train_indices & set(audit["calibration_indices"]))
            scores = pl.read_csv(root / "full/patient_scores.csv")
            for split in ("train", "tuning"):
                self.assertEqual(
                    scores.filter(pl.col("split") == split)["subject_id"].n_unique(), 20
                )
            for cohort, count in (("random", 16), ("uncommon", 4)):
                destination = root / "full/cohorts" / cohort
                self.assertTrue((destination / "summary.txt").is_file())
                rows = pl.read_csv(destination / "patient_losses.csv")
                self.assertEqual(rows.height, 2 * count)
                self.assertEqual(rows["sampling_cohort"].unique().to_list(), [cohort])
            self.assertTrue((root / "full/audit.txt").is_file())
            from bonsai.run.reanalyze_memorization import reanalyze

            reanalyze(root / "full", root / "review", 4, minimum_diversity=2)
            self.assertTrue((root / "review/cohorts/random/summary.txt").is_file())
            self.assertTrue(
                (
                    root / "review/diversity_filtered/cohorts/uncommon/summary.txt"
                ).is_file()
            )
            reviewed = json.loads((root / "review/run_manifest.json").read_text())
            self.assertEqual(
                set(reviewed["cohort_statuses"]), {"random", "uncommon", "combined"}
            )
            from bonsai.run.inspect_memorization import examples

            self.assertTrue(list(examples(root / "review/cohorts/random", None)))
            filtered_examples = list(
                examples(root / "review/diversity_filtered/cohorts/random", None)
            )
            self.assertTrue(filtered_examples)
            self.assertTrue(
                all(
                    row["sampling_cohort"] == "random"
                    and row["scores"]["unique_reference_tokens"] >= 2
                    for row in filtered_examples
                )
            )


if __name__ == "__main__":
    unittest.main()
