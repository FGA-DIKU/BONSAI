"""Original memorization method: autoregressive feedback, scores, loss and outputs."""

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import torch
from torch import nn

from bonsai.functional.features import compute_abspos
from bonsai.functional.memorization_cache import (
    build_pools,
    read_cache,
    save_cache,
    source_stamp,
)
from bonsai.functional.memorization_inspection import PatientInspection
from bonsai.functional.subject_data import prepare_subject_data
from bonsai.modules.datasets.PretrainDataset import ARPretrainDataset, PretrainDataset
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain, BonsaiValuePretrain
from bonsai.run import memorization_check as check


def patient(identifier, codes, values=False):
    length = len(codes)
    row = {
        "subject_id": identifier,
        "code": torch.tensor(codes),
        "age": torch.arange(length, dtype=torch.float),
        "abspos": torch.arange(length, dtype=torch.float),
        "segment": torch.arange(1, length + 1),
    }
    if values:
        row["numeric_value"] = torch.arange(length, dtype=torch.float)
    return row


def model(values=False):
    settings = {
        "vocab_size": 8,
        "max_seqlen": 128,
        "hidden_size": 64,
        "num_layers": 1,
        "bias": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "causal": True,
        "attn_type": "sdpa",
    }
    if values:
        settings["value_embedding_mode"] = "film"
    return (BonsaiValuePretrain if values else BonsaiPretrain)(**settings).eval()


def save_checkpoint(path, network):
    torch.save(
        {
            "hyper_parameters": network.hparams,
            "state_dict": {
                f"model.{key}": value for key, value in network.state_dict().items()
            },
        },
        path,
    )


class TestMemorizationCheck(unittest.TestCase):
    def test_configurable_diversity_threshold_and_low_diversity_denominators(self):
        scores = pl.DataFrame(
            [
                {
                    "mode": "true_features",
                    "split": "train",
                    "X": 32,
                    "Y": 32,
                    "exact_match": exact,
                    "overlap": 1.0,
                    "rare_codes": float("nan"),
                    "unique_predicted_tokens": unique,
                }
                for unique, exact in ((1, 1.0), (2, 0.0), (3, 0.0))
            ]
        )
        row = check.patient_flag_rates(scores, 3).row(0, named=True)
        self.assertEqual(row["patients_low_diversity"], 2)
        self.assertEqual(row["patients_flagged"], 2)
        self.assertEqual(row["patients_flagged_low_diversity"], 1)
        self.assertEqual(row["percent_low_diversity_among_exact"], 100)
        self.assertAlmostEqual(row["percent_low_diversity"], 200 / 3)
        stricter = check.patient_flag_rates(scores, 4).row(0, named=True)
        self.assertEqual(stricter["patients_low_diversity"], 3)
        self.assertEqual(stricter["patients_flagged"], 1)
        tails = check.patient_tails(scores, 4)
        self.assertTrue(
            check.inspection_flags(tails)
            .filter(pl.col("score") == "overlap")
            .is_empty()
        )

    def test_overlap_flags_require_three_unique_predicted_clinical_tokens(self):
        records = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter = PatientInspection(
                root, {f"TOKEN_{i}": i for i in range(8)}, 2, 32, 0.9
            )
            for unique in (2, 3):
                prediction = np.resize(np.arange(1, unique + 1), 32)
                truth = np.roll(prediction, 1)
                metrics = check.score_patient(prediction, truth, {1, 2, 3}, set())
                self.assertEqual(metrics["overlap"], 1)
                self.assertEqual(metrics["exact_match"], 0)
                window = patient(unique, [1] * 32 + truth.tolist())
                exporter.record(
                    window,
                    prediction,
                    "train",
                    "forward_fill",
                    32,
                    32,
                    metrics,
                    False,
                    7,
                )
                records.append(
                    {
                        "mode": "forward_fill",
                        "split": "train",
                        "X": 32,
                        "Y": 32,
                        **metrics,
                    }
                )
            exporter.close()
            flags = [
                json.loads(line)
                for line in (root / "flagged_trajectories.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([row["subject_id"] for row in flags], [3])
            from bonsai.run.inspect_memorization import format_example

            self.assertIn("Decoded prompt: TOKEN_1 | TOKEN_1", format_example(flags[0]))
        scores = pl.DataFrame(records)
        rates = check.patient_flag_rates(scores).row(0, named=True)
        self.assertEqual(rates["patients_overlap_flagged"], 1)
        self.assertEqual(rates["percent_flagged"], 50)
        tails = check.patient_tails(scores)
        overlap = (
            check.inspection_flags(tails)
            .filter(pl.col("score") == "overlap")
            .row(0, named=True)
        )
        self.assertEqual(overlap["patients_high_overlap_diverse"], 1)
        # Special tokens do not satisfy the clinical-token diversity requirement.
        metrics = check.score_patient(
            np.array([1, 2, 7]), np.array([2, 1, 7]), {1, 2}, set()
        )
        self.assertEqual(metrics["unique_predicted_tokens"], 2)

    def test_patient_flag_percentages_use_cell_denominators_without_double_counting(
        self,
    ):
        scores = pl.DataFrame(
            [
                {
                    "mode": "true_features",
                    "split": "train",
                    "X": 32,
                    "Y": horizon,
                    "exact_match": exact,
                    "overlap": overlap,
                    "rare_codes": float("nan"),
                }
                for horizon in (16, 32)
                for exact, overlap in ((1.0, 1.0), (0.0, 0.95), (0.0, float("nan")))
            ]
        )
        rates = check.patient_flag_rates(scores)
        row = rates.filter(pl.col("Y") == 32).row(0, named=True)
        self.assertEqual(row["patients"], 3)
        self.assertEqual(row["patients_flagged"], 2)
        self.assertEqual(row["patients_exact_flagged"], 1)
        self.assertEqual(row["patients_overlap_flagged"], 2)
        self.assertAlmostEqual(row["percent_flagged"], 200 / 3)
        self.assertIsNone(rates.filter(pl.col("Y") == 16)["percent_flagged"][0])
        tails = check.patient_tails(scores)
        exact = tails.filter((pl.col("Y") == 32) & (pl.col("score") == "exact_match"))
        self.assertAlmostEqual(exact["percent_scoring_1"][0], 100 / 3)

    def test_inspection_exports_flagged_ids_histories_predictions_and_controls(self):
        network = nn.Linear(1, 1)
        network.pretrain_head = nn.Identity()

        def hidden(_, batch):
            logits = torch.zeros((*batch["code"].shape, 8))
            logits[..., 3] = 10
            return logits

        rows = [
            patient(101, [1] * 32 + [7, 3] * 16, values=True),
            patient(202, [1] * 32 + [7, 4] * 16, values=True),
            patient(303, [1] * 32 + [7] * 32, values=True),
        ]
        rows[0]["numeric_value"][:] = float("nan")
        vocabulary = {f"T{i}": i for i in range(8)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inspection = PatientInspection(root, vocabulary, 1, 32, 0.9)
            with patch.object(check, "hidden_states", side_effect=hidden):
                scores = check.score_all_patients(
                    network,
                    {"train": rows},
                    [32],
                    [32],
                    3,
                    set(range(1, 7)),
                    {3},
                    mode="true_features",
                    sep_token_id=7,
                    supply_sep=True,
                    inspection=inspection,
                )
            inspection.close()
            flags = [
                json.loads(line)
                for line in (root / "flagged_trajectories.jsonl")
                .read_text()
                .splitlines()
            ]
            controls = [
                json.loads(line)
                for line in (root / "unflagged_examples.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["subject_id"] for row in flags], [101])
            self.assertEqual([row["subject_id"] for row in controls], [202])
            flag = flags[0]
            self.assertEqual(flag["prompt"]["ids"], [1] * 32)
            self.assertEqual(flag["true_continuation"]["ids"], [7, 3] * 16)
            self.assertEqual(flag["predicted_continuation"]["ids"], [7, 3] * 16)
            self.assertEqual(flag["supplied_sep_positions"], list(range(0, 32, 2)))
            self.assertEqual(flag["scored_positions"], 16)
            self.assertEqual(flag["prompt_features"]["numeric_value"], [None] * 32)
            from bonsai.run.inspect_memorization import examples, format_example

            loaded = list(examples(root, "101", "true_features", 32, 32))
            self.assertEqual(len(loaded), 1)
            self.assertIn("Patient 101", format_example(loaded[0]))
            self.assertIn("supplied_SEP", format_example(loaded[0]))
            formatted = format_example(loaded[0], width=80)
            self.assertTrue(all(len(line) <= 80 for line in formatted.splitlines()))
            self.assertIn("LOW DIVERSITY", formatted)
            self.assertIn("CONTINUATION", formatted)
            self.assertIn("TRUE CODE", formatted)
            self.assertIn("PREDICTED CODE", formatted)
            self.assertIn("numeric_value=missing", formatted)
            self.assertLess(
                formatted.index("CONTINUATION"), formatted.index("PROMPT HISTORY")
            )
            # Existing exports can be reformatted without loading a model/cache.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bonsai.run.inspect_memorization",
                    "--report-dir",
                    str(root),
                    "--output-file",
                    str(root / "audit.txt"),
                    "--width",
                    "80",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = (root / "audit.txt").read_text()
            self.assertIn("Patient 101", audit)
            self.assertIn("Patient 202", audit)
            self.assertNotIn("Patient 303", audit)
            self.assertIn("CASE INDEX", audit)
            self.assertIn("REVIEW NOTE", audit)
            self.assertIn("{{{1", audit)
            self.assertIn("foldmethod=marker", audit)
            self.assertEqual(list(examples(root, "999")), [])
            exported = pl.read_csv(root / "patient_scores.csv")
            self.assertEqual(exported["subject_id"].to_list(), [101, 202])
            tail = check.patient_tails(scores).filter(pl.col("score") == "exact_match")
            self.assertEqual(tail["patients_scoring_1"].sum(), len(flags))

    def test_evaluation_cache_preserves_pools_rarity_and_nested_samples(self):
        records = [
            patient(i, [1, 7, 2, 3, 7, 4] if i % 2 else [1, 2, 3]) for i in range(20)
        ]
        windows = {"train": check.training_windows(records, 8, 1)}
        pools = build_pools(windows, [3, 6], 5, check.sample_patients, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            signature = {"context": 8, "seed": 0}
            save_cache(path, signature, 5, pools, {1, 2, 3}, {3})
            cached, countable, rare, capacity = read_cache(path, signature, 5)
            self.assertEqual((countable, rare, capacity), ({1, 2, 3}, {3}, 5))
            for minimum in (3, 6):
                expected = check.sample_patients(windows["train"], minimum, 5)
                actual = check.sample_patients(cached["train"], minimum, 5)
                self.assertEqual(
                    [w["subject_id"] for w in actual],
                    [w["subject_id"] for w in expected],
                )
                small = check.sample_patients(cached["train"], minimum, 2)
                self.assertTrue(
                    {w["subject_id"] for w in small}
                    <= {w["subject_id"] for w in actual}
                )
            for minimum in (3, 6):
                self.assertTrue(
                    all(
                        len(w["code"]) >= minimum
                        for w in check.sample_patients(cached["train"], minimum, 5)
                    )
                )
            with self.assertRaisesRegex(ValueError, "settings/source"):
                read_cache(path, {"context": 9}, 5)
            with self.assertRaisesRegex(ValueError, "capacity"):
                read_cache(path, signature, 6)
            with self.assertRaises(FileExistsError):
                save_cache(path, signature, 5, pools, set(), set())
            self.assertEqual(read_cache(path, signature, 5)[2], {3})
            self.assertEqual(source_stamp(path)["size"], path.stat().st_size)

    def test_cli_reuses_cache_and_matches_uncached_full_sample_with_sep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(root / "model.ckpt", model())
            torch.save(
                {"[PAD]": 0, "[SEP]": 7, **{f"C{i}": i for i in range(1, 7)}},
                root / "vocabulary.pt",
            )
            for split, offset in (("train", 0), ("tuning", 10)):
                torch.save(
                    [patient(offset + i, [1, 2, 7, 3, 4, 7, 5, 6]) for i in range(4)],
                    root / f"subject_data_{split}.pt",
                )
            (root / "config.yaml").write_text(
                "paths:\n  population: null\ntraining:\n  cutoff_date: null\n  max_len: 128\n"
            )
            command = [
                sys.executable,
                "-m",
                "bonsai.run.memorization_check",
                "--sampling-design",
                "legacy",
                "--checkpoint",
                str(root / "model.ckpt"),
                "--data-dir",
                str(root),
                "--run-config",
                str(root / "config.yaml"),
                "--device",
                "cpu",
                "--prompt-events",
                "2",
                "--continuation-events",
                "1",
                "4",
                "--min-patients",
                "2",
                "--sep-excluded-from-training-loss",
                "--export-patient-inspection",
            ]
            for name, maximum, cached in (
                ("preflight", 2, True),
                ("full", 3, True),
                ("baseline", 3, False),
            ):
                options = (
                    [
                        "--evaluation-cache",
                        str(root / "cache.pt"),
                        "--cache-patients",
                        "3",
                    ]
                    if cached
                    else []
                )
                result = subprocess.run(
                    [
                        *command,
                        "--max-patients",
                        str(maximum),
                        "--output-dir",
                        str(root / name),
                        *options,
                    ],
                    capture_output=True,
                    check=False,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                if name == "full":
                    self.assertIn("Loading sampled evaluation cache", result.stdout)
                    self.assertNotIn("Loading train metadata", result.stdout)
                    self.assertNotIn("Counting code frequencies", result.stdout)
                    manifest = json.loads(
                        (root / name / "run_manifest.json").read_text()
                    )
                    self.assertTrue(manifest["evaluation_cache"]["reused"])
            for report in (
                "excess.csv",
                "loss_check.csv",
                "patient_tails.csv",
                "coverage.csv",
                "patient_flag_rates.csv",
            ):
                from polars.testing import assert_frame_equal

                full, baseline = [
                    pl.read_csv(root / name / report) for name in ("full", "baseline")
                ]
                assert_frame_equal(
                    full.sort(full.columns), baseline.sort(baseline.columns)
                )
            self.assertTrue((root / "full/patient_losses.csv").is_file())
            self.assertTrue((root / "full/audit.txt").is_file())
            self.assertIn("BONSAI PATIENT AUDIT", (root / "full/audit.txt").read_text())
            self.assertEqual(pl.read_csv(root / "full/patient_scores.csv").height, 30)
            self.assertTrue((root / "full/unflagged_examples.jsonl").read_text())
            from bonsai.run.reanalyze_memorization import reanalyze

            reanalyze(
                root / "full",
                root / "review",
                10,
                minimum_diversity=2,
                diversity_basis="reference",
            )
            review = pl.read_csv(root / "review/patient_flag_rates.csv")
            self.assertEqual(
                review["patients_low_diversity"].sum(), review["patients"].sum()
            )
            metadata = json.loads((root / "review/run_manifest.json").read_text())
            self.assertEqual(
                metadata["inspection_triggers"][
                    "minimum_unique_predicted_clinical_tokens_for_overlap"
                ],
                10,
            )
            self.assertEqual(
                metadata["reanalysis"]["source_report"], str((root / "full").resolve())
            )
            self.assertTrue((root / "review/patient_scores.csv").is_file())
            from bonsai.run.inspect_memorization import examples

            self.assertTrue(list(examples(root / "review", None)))
            filtered_dir = root / "review/diversity_filtered"
            filtered_scores = pl.read_csv(filtered_dir / "patient_scores.csv")
            self.assertTrue((filtered_scores["unique_reference_tokens"] >= 2).all())
            self.assertEqual(filtered_scores.height, 18)
            exclusions = pl.read_csv(filtered_dir / "diversity_exclusions.csv")
            self.assertEqual(exclusions["patients_excluded"].sum(), 12)
            self.assertIn(
                "loss/AUC reused without filtering",
                (filtered_dir / "summary.txt").read_text(),
            )
            reanalyze(root / "full", root / "empty_review", 3, minimum_diversity=100)
            self.assertEqual(
                pl.read_csv(root / "empty_review/diversity_filtered/coverage.csv")[
                    "patients"
                ].sum(),
                0,
            )
            # Plots are generated after completion, with no model/cache access.
            import importlib.util

            if importlib.util.find_spec("matplotlib"):
                from bonsai.run.plot_memorization import plot_reports

                plot_reports(root / "full", root / "figures")
                figures = {path.name for path in (root / "figures").glob("*.png")}
                self.assertTrue(
                    {
                        "loss_auc.png",
                        "coverage.png",
                        "true_features_metrics.png",
                        "true_features_denominators.png",
                        "true_features_score_tails.png",
                        "true_features_exact_diversity.png",
                    }
                    <= figures
                )
            # A source change must reject reuse without silently rebuilding.
            (root / "subject_data_train.pt").touch()
            rejected = subprocess.run(
                [
                    *command,
                    "--max-patients",
                    "3",
                    "--output-dir",
                    str(root / "stale"),
                    "--evaluation-cache",
                    str(root / "cache.pt"),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("settings/source", rejected.stderr)

    def test_supplied_sep_is_appended_but_future_clinical_codes_are_not(self):
        network = nn.Linear(1, 1)
        network.pretrain_head = nn.Identity()
        seen = []

        def hidden(_, batch):
            seen.append(batch["code"].tolist()[0])
            logits = torch.zeros((*batch["code"].shape, 8))
            logits[..., 3] = 10
            return logits

        window = patient(1, [1, 2, 7, 5, 6])
        with patch.object(check, "hidden_states", side_effect=hidden):
            result = check.generate(
                network,
                [window],
                2,
                3,
                mode="true_features",
                sep_token_id=7,
                supply_sep=True,
            )
        np.testing.assert_array_equal(result, [[7, 3, 3]])
        self.assertEqual(seen, [[1, 2], [1, 2, 7], [1, 2, 7, 3]])
        with patch.object(check, "hidden_states", side_effect=hidden):
            scores = check.score_all_patients(
                network,
                {"train": [window]},
                [2],
                [1, 3],
                1,
                set(range(1, 7)),
                set(),
                mode="true_features",
                sep_token_id=7,
                supply_sep=True,
            )
        self.assertEqual(scores["Y"].to_list(), [3])
        self.assertEqual(scores["exact_match"].to_list(), [0.0])

    def test_loss_excludes_sep_targets_and_sampling_omits_sep_only_records(self):
        network = model()
        window = patient(1, [1, 7, 2, 7, 3])
        batch = check.make_batch([window], 4, "cpu")
        with torch.inference_mode():
            logits = network.pretrain_head(check.hidden_states(network, batch))[0]
            targets = window["code"][1:]
            expected = torch.nn.functional.cross_entropy(
                logits[targets != 7], targets[targets != 7]
            ).item()
        actual = check.patient_losses(network, [window], exclude_target=7)
        self.assertAlmostEqual(actual[0], expected, places=5)
        selected = check.sample_patients(
            [patient(2, [1, 7, 7]), window],
            2,
            10,
            require_targets=True,
            exclude_target=7,
        )
        self.assertEqual([row["subject_id"] for row in selected], [1])

    def test_runtime_projection_scales_inference_but_not_setup(self):
        timing = check.runtime_estimate(
            {
                "setup_seconds": 120,
                "generation_scoring_seconds": 60,
                "loss_seconds": 30,
            },
            sample_cap=100,
        )
        self.assertEqual(timing["estimated_target_seconds"], 120 + 1000 * 90)
        self.assertEqual(timing["target_patients_per_group"], 100000)

    def test_rarity_scan_does_not_copy_or_normalize_windows(self):
        raw = [patient(1, [1, 2, 3, 4, 5, 6])]
        lazy = check.training_windows(raw, 3, 1)
        vocabulary = {"[PAD]": 0, **{f"C{i}": i for i in range(1, 8)}}
        expected = check.code_groups(vocabulary, [lazy[0]])
        with patch.object(
            check.TrainingWindows,
            "__getitem__",
            side_effect=AssertionError("full window materialized"),
        ):
            actual = check.code_groups(vocabulary, lazy)
        self.assertEqual(actual, expected)

    def test_pipeline_parquet_records_support_all_modes_and_value_models(self):
        """Exercise the real parquet -> subject_data -> mapped inference path."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, offset in (("train", 0), ("tuning", 10)):
                shard = root / split
                shard.mkdir()
                rows = []
                for identifier, length in enumerate((3, 9, 40), start=offset):
                    for position in range(length):
                        rows.append(
                            {
                                "subject_id": identifier,
                                "code": 1 + position % 7,
                                "age": 50.0 + position / 365,
                                "abspos": float(position // 2),
                                "segment": 1 + position // 2,
                                "numeric_value": 1.25 if position % 3 == 0 else None,
                            }
                        )
                pl.DataFrame(rows).cast(
                    {
                        "code": pl.Int64,
                        "age": pl.Float32,
                        "abspos": pl.Float32,
                        "segment": pl.Int32,
                        "numeric_value": pl.Float32,
                    }
                ).write_parquet(shard / "0.parquet")
                torch.save(
                    prepare_subject_data(shard), root / f"subject_data_{split}.pt"
                )
            windows = check.load_windows(root, ("train", "tuning"), context=16)
            vocabulary = {"[PAD]": 0, **{f"CODE_{i}": i for i in range(1, 8)}}
            for values in (False, True):
                network = model(values)
                check.validate_windows(windows, vocabulary, network)
                selected = check.sample_patients(windows["train"], 8, 10)
                self.assertEqual(sorted(len(row["code"]) for row in selected), [9, 17])
                self.assertTrue(torch.isnan(selected[0]["numeric_value"]).any())
                for mode in check.MODES:
                    generated = check.generate(network, selected, 4, 4, mode=mode)
                    self.assertEqual(generated.shape, (2, 4))
                conditional = check.generate(
                    network,
                    selected,
                    4,
                    4,
                    mode="true_features",
                    supply_sep=True,
                    sep_token_id=7,
                )
                truth = torch.stack([row["code"][4:8] for row in selected]).numpy()
                np.testing.assert_array_equal(
                    conditional[truth == 7], truth[truth == 7]
                )
                self.assertTrue(
                    np.isfinite(
                        check.patient_losses(network, selected, exclude_target=7)
                    ).all()
                )
                losses = check.patient_losses(network, selected, batch_size=2)
                self.assertTrue(np.isfinite(losses).all())

    def test_load_model_and_training_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            for values in (False, True):
                original = model(values)
                path = Path(directory) / "model.ckpt"
                save_checkpoint(path, original)
                loaded = check.load_model(path, "cpu")
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)
        subjects = [patient(1, [1, 2, 3, 4, 5, 6, 7, 2])]
        window = check.training_windows(subjects, 4, 1)[0]
        expected = ARPretrainDataset(subjects, max_len=4, background_length=1)[0]
        torch.testing.assert_close(window["code"][:-1], expected["code"])
        torch.testing.assert_close(window["code"][1:], expected["target"])

    def test_lazy_windows_match_original_cutoff_and_truncation(self):
        original = patient(1, list(range(1, 13)), values=True)
        original["segment"] = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        before = compute_abspos(datetime(2014, 1, 1))
        after = compute_abspos(datetime(2016, 1, 1))
        original["abspos"] = torch.tensor([before] * 6 + [after] * 6)
        for cutoff in (
            None,
            {"year": 2015, "month": 1, "day": 1},
            {"year": 2013, "month": 1, "day": 1},
        ):
            expected = PretrainDataset([original], 5, 2, cutoff)[0]
            lazy = check.training_windows([original], 4, 2, cutoff)
            self.assertNotIsInstance(lazy, list)
            actual = lazy[0]
            for stop in (0, 1, 3, 20):
                torch.testing.assert_close(
                    lazy.prefix_codes(0, stop), actual["code"][:stop]
                )
            self.assertEqual(int(lazy.lengths[0]), len(expected["code"]))
            for field, value in expected.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[field], value, equal_nan=True)
                else:
                    self.assertEqual(actual[field], value)
        self.assertEqual(original["code"].tolist(), list(range(1, 13)))

    def test_reservoir_materializes_only_selected_windows_and_is_repeatable(self):
        windows = check.training_windows(
            [patient(i, [1, 2, 3, 4, 5, 6]) for i in range(100)], 4, 1
        )
        accesses = []
        get_window = check.TrainingWindows.__getitem__

        def counted(dataset, index):
            accesses.append(index)
            return get_window(dataset, index)

        with patch.object(check.TrainingWindows, "__getitem__", counted):
            first = check.sample_patients(windows, 3, 5)
            self.assertEqual(len(accesses), 5)
            second = check.sample_patients(windows, 3, 5, require_targets=True)
            self.assertEqual(len(accesses), 10)
        self.assertEqual(
            [w["subject_id"] for w in first], [w["subject_id"] for w in second]
        )
        self.assertEqual(len({w["subject_id"] for w in first}), 5)

    def test_long_histories_are_sliced_before_any_clone(self):
        original = patient(1, [1] * 100_000, values=True)
        windows = check.training_windows([original], 4, 1)
        clone = torch.Tensor.clone
        cloned_sizes = []

        def bounded_clone(tensor, *args, **kwargs):
            cloned_sizes.append(tensor.numel())
            return clone(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "clone", bounded_clone):
            selected = windows[0]
        self.assertTrue(all(size <= 5 for size in cloned_sizes))
        self.assertEqual(
            selected["code"].untyped_storage().nbytes(),
            5 * selected["code"].element_size(),
        )
        selected["code"][0] = 7
        self.assertEqual(original["code"][0].item(), 1)

    def test_mapped_files_and_legacy_files_do_not_trigger_eager_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subjects.pt"
            torch.save([patient(1, [1, 2, 3])], path)
            records = check.load_mapped(path)
            window = check.training_windows(records, 2, 1)[0]
            window["code"][0] = 7
            self.assertEqual(records[0]["code"][0].item(), 1)
            torch.save(
                records,
                Path(directory) / "legacy.pt",
                _use_new_zipfile_serialization=False,
            )
            with self.assertRaisesRegex(RuntimeError, "full-RAM fallback"):
                check.load_mapped(Path(directory) / "legacy.pt")

    def test_mapped_loading_filters_each_split_before_retaining_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                [patient(i, [1, 2, 3]) for i in (1, 2, 3)],
                root / "subject_data_train.pt",
            )
            torch.save(
                [patient(i, [1, 2, 3]) for i in (10, 11, 3)],
                root / "subject_data_tuning.pt",
            )
            population = root / "population.csv"
            population.write_text("subject_id\n1\n2\n10\n11\n")
            windows = check.load_windows(root, ("train", "tuning"), 4, population)
            self.assertTrue(
                all(
                    isinstance(value, check.TrainingWindows)
                    for value in windows.values()
                )
            )
            self.assertEqual(
                [r["subject_id"] for r in windows["train"].subjects], [1, 2]
            )
            self.assertEqual(
                [r["subject_id"] for r in windows["tuning"].subjects], [10, 11]
            )

    def test_generation_feeds_back_predictions_and_retains_real_features(self):
        network = nn.Linear(1, 1)
        network.pretrain_head = nn.Identity()
        contexts = []
        features = []

        def hidden(_, batch):
            contexts.append(batch["code"][0].tolist())
            features.append(batch["numeric_value"][0].tolist())
            return torch.nn.functional.one_hot(batch["code"] + 1, num_classes=8).float()

        window = patient(1, [1, 2, 7, 7, 7, 7], values=True)
        with patch.object(check, "hidden_states", side_effect=hidden):
            generated = check.generate(network, [window], 2, 4, mode="true_features")
        self.assertEqual(contexts, [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(generated, [[3, 4, 5, 6]])
        self.assertEqual(features[-1], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(window["code"].tolist(), [1, 2, 7, 7, 7, 7])

    def test_forward_fill_ignores_future_features_and_fills_missing_values(
        self,
    ):
        network = nn.Linear(1, 1)
        network.pretrain_head = nn.Identity()
        seen = []

        def hidden(_, batch):
            seen.append({key: value.clone() for key, value in batch.items()})
            return torch.nn.functional.one_hot(batch["code"] + 1, num_classes=8).float()

        original = patient(1, [1, 2, 7, 7, 7, 7], values=True)
        changed = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in original.items()
        }
        for field in check.FIELDS:
            changed[field][2:] = 100
        forward_fill = {
            key: value[:2] if isinstance(value, torch.Tensor) else value
            for key, value in original.items()
        }
        with patch.object(check, "hidden_states", side_effect=hidden):
            outputs = [
                check.generate(
                    network,
                    [window],
                    2,
                    4,
                    mode="forward_fill",
                    sep_token_id=4,
                )
                for window in (original, changed, forward_fill)
            ]
        for output in outputs:
            np.testing.assert_array_equal(output, [[3, 4, 5, 6]])
        self.assertTrue(torch.isnan(seen[1]["numeric_value"][0, -1]))
        self.assertTrue(
            torch.isnan(seen[2]["numeric_value"][0, -1])
        )  # SEP has no value.
        self.assertEqual(seen[2]["segment"][0, -1].item(), 2)  # SEP closes old segment.
        self.assertEqual(
            seen[3]["segment"][0, -1].item(), 3
        )  # Following event advances.
        self.assertEqual(seen[3]["age"][0, -1].item(), 1.0)
        self.assertEqual(seen[3]["abspos"][0, -1].item(), 1.0)

    def test_real_models_forward_fill_decoding_is_independent_of_future_features(self):
        torch.manual_seed(2)
        original = patient(1, [1, 2, 7, 7, 7, 7], values=True)
        changed = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in original.items()
        }
        for field in check.FIELDS:
            changed[field][2:] = 6
        for values in (False, True):
            network = model(values)
            first = check.generate(
                network,
                [original],
                2,
                4,
                mode="forward_fill",
                sep_token_id=4,
            )
            second = check.generate(
                network,
                [changed],
                2,
                4,
                mode="forward_fill",
                sep_token_id=4,
            )
            np.testing.assert_array_equal(first, second)

    def test_code_only_bypasses_all_non_code_embeddings_for_both_models(self):
        torch.manual_seed(3)
        window = {"subject_id": 1, "code": torch.tensor([1, 2])}
        for values in (False, True):
            network = model(values)
            with patch.object(
                network.embeddings,
                "forward",
                side_effect=AssertionError("Metadata embeddings must not run"),
            ):
                generated = check.generate(network, [window], 2, 4, mode="code_only")
            # Independent reference: code lookup -> unchanged transformer -> code head.
            codes = window["code"][None].clone()
            expected = []
            with torch.inference_mode():
                for _ in range(4):
                    embedded = network.drop(network.embeddings.code_embedding(codes))
                    last = network.forward_sdpa(embedded, attn_mask=None)[:, -1]
                    next_code = network.pretrain_head(last).argmax(-1, keepdim=True)
                    expected.append(next_code.item())
                    codes = torch.cat([codes, next_code], dim=1)
            np.testing.assert_array_equal(generated, [expected])

    def test_original_multiset_overlap_and_rare_scores(self):
        scores = check.score_patient(np.array([2, 3]), np.array([2, 2]), {2, 3}, {2})
        self.assertEqual(
            scores,
            {
                "exact_match": 0.0,
                "overlap": 0.5,
                "rare_codes": 0.5,
                "unique_predicted_tokens": 2,
                "unique_reference_tokens": 1,
            },
        )
        scores = check.score_patient(np.array([2, 3]), np.array([3, 2]), {2, 3}, {2})
        self.assertEqual(
            scores,
            {
                "exact_match": 0.0,
                "overlap": 1.0,
                "rare_codes": 1.0,
                "unique_predicted_tokens": 2,
                "unique_reference_tokens": 2,
            },
        )

    def test_grid_uses_one_continuation_per_context_and_same_patients_across_horizons(
        self,
    ):
        windows = [patient(1, [1] * 8), patient(2, [1] * 8)]
        with patch.object(
            check, "generate", return_value=np.ones((2, 2), dtype=int)
        ) as decode:
            scores = check.score_all_patients(
                None, {"train": windows}, [2, 4], [1, 2], 10, {1}, {1}
            )
        self.assertEqual(decode.call_count, 2)
        self.assertEqual(scores.height, 8)
        self.assertEqual(
            set(zip(scores["X"], scores["Y"])), {(2, 1), (2, 2), (4, 1), (4, 2)}
        )
        self.assertEqual(scores["exact_match"].to_list(), [1.0] * 8)

    def test_loss_matches_direct_next_code_cross_entropy(self):
        torch.manual_seed(1)
        windows = [patient(1, [1, 2, 3, 4], True), patient(2, [1, 5], True)]
        for values in (False, True):
            network = model(values)
            expected = []
            with torch.inference_mode():
                for window in windows:
                    batch = check.make_batch([window], len(window["code"]) - 1, "cpu")
                    logits = network.pretrain_head(check.hidden_states(network, batch))[
                        0
                    ]
                    expected.append(
                        torch.nn.functional.cross_entropy(
                            logits, window["code"][1:]
                        ).item()
                    )
            np.testing.assert_allclose(
                check.patient_losses(network, windows, 2), expected, rtol=1e-5
            )

    def test_loss_head_is_chunked_without_changing_cross_entropy(self):
        network = model()
        sizes = []
        hook = network.pretrain_head.register_forward_pre_hook(
            lambda _, args: sizes.append(args[0].shape[0])
        )
        self.addCleanup(hook.remove)
        window = patient(1, [2] * 601)
        with patch.object(check, "hidden_states", return_value=torch.zeros(1, 600, 64)):
            losses = check.patient_losses(network, [window])
        self.assertEqual(sizes, [512, 88])
        np.testing.assert_allclose(losses, [np.log(8)], rtol=1e-6)

    def test_short_horizons_keep_short_patients(self):
        windows = [patient(1, [1] * 3), patient(2, [1] * 6)]

        def decode(_, part, prompt, steps, batch_size, **kwargs):
            return np.ones((len(part), steps), dtype=int)

        with patch.object(check, "generate", side_effect=decode):
            scores = check.score_all_patients(
                None, {"train": windows}, [2], [1, 4], 10, {1}, {1}
            )
        self.assertEqual(scores.filter(pl.col("Y") == 1).height, 2)
        self.assertEqual(scores.filter(pl.col("Y") == 4).height, 1)

    def test_empty_grid_is_inconclusive_and_auc_bounds_are_valid(self):
        scores = check.score_all_patients(
            None, {"train": []}, [32], [32], 10, {1}, set()
        )
        excess = check.excess_over_unseen(scores, "train", "tuning")
        loss = check.membership_from_loss(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        self.assertGreaterEqual(loss["auc_lower"][0], 0.0)
        self.assertLessEqual(loss["auc_upper"][0], 1.0)
        coverage = check.grid_coverage(scores, [32], [32], ("train", "tuning"), 128, 2)
        self.assertIn(
            "INCONCLUSIVE", check.comparison_summary(excess, loss, 2, None, coverage)
        )

    def test_patient_tails_expose_perfect_reproduction_hidden_by_mean(self):
        scores = pl.DataFrame(
            [
                {
                    "mode": "forward_fill",
                    "split": "train",
                    "X": 32,
                    "Y": 32,
                    "exact_match": value,
                    "overlap": value,
                    "rare_codes": float("nan"),
                }
                for value in [0.0, 0.0, 0.0, 1.0]
            ]
        )
        tails = (
            check.patient_tails(scores)
            .filter(pl.col("score") == "exact_match")
            .row(0, named=True)
        )
        self.assertEqual(tails["median"], 0.0)
        self.assertEqual(tails["max"], 1.0)
        self.assertEqual(tails["patients_scoring_1"], 1)
        self.assertEqual(tails["patients"], 4)

    def test_one_reproduced_patient_is_flagged_even_when_means_are_underpowered(self):
        rows = []
        for split in ("train", "tuning"):
            for i in range(8):
                value = float(split == "train" and i == 0)
                rows.append(
                    {
                        "mode": "forward_fill",
                        "split": split,
                        "X": 32,
                        "Y": 32,
                        "exact_match": value,
                        "overlap": value,
                        "rare_codes": float("nan"),
                    }
                )
        scores = pl.DataFrame(rows)
        excess = check.excess_over_unseen(scores, "train", "tuning")
        loss = check.membership_from_loss(np.ones(8), np.ones(8))
        tails = check.patient_tails(scores)
        coverage = check.grid_coverage(scores, [32], [32], ("train", "tuning"), 128, 50)
        summary = check.comparison_summary(excess, loss, 50, tails, coverage)
        self.assertTrue(summary.startswith("INCONCLUSIVE:"))
        self.assertEqual(check.inspection_flags(tails).height, 2)
        self.assertIn("Patient-level reproduction triggers: 2 cells", summary)
        self.assertIn(
            "They do not independently determine this comparison gate", summary
        )

    def test_passing_requires_all_modes_metrics_and_enough_patients(self):
        scores = pl.DataFrame(
            [
                {
                    "mode": mode,
                    "split": split,
                    "X": 32,
                    "Y": 32,
                    "exact_match": 0.0,
                    "overlap": 0.0,
                    "rare_codes": 0.0,
                }
                for mode in check.MODES
                for split in ("train", "tuning")
                for _ in range(8)
            ]
        )
        excess = check.excess_over_unseen(scores, "train", "tuning")
        loss = check.membership_from_loss(np.ones(8), np.ones(8))
        tails = check.patient_tails(scores)
        coverage = check.grid_coverage(scores, [32], [32], ("train", "tuning"), 128, 4)
        self.assertTrue(
            check.comparison_summary(excess, loss, 4, tails, coverage).startswith(
                "CHECKS PASSED:"
            )
        )
        self.assertTrue(
            check.comparison_summary(excess, loss, 50, tails, coverage).startswith(
                "INCONCLUSIVE:"
            )
        )
        only_primary = excess.filter(pl.col("mode") == "forward_fill")
        self.assertTrue(
            check.comparison_summary(only_primary, loss, 4, tails, coverage).startswith(
                "INCONCLUSIVE:"
            )
        )

    def test_incomplete_requested_grid_cannot_pass(self):
        scores = pl.DataFrame(
            [
                {
                    "mode": mode,
                    "split": split,
                    "X": 32,
                    "Y": 32,
                    "exact_match": 0.0,
                    "overlap": 0.0,
                    "rare_codes": 0.0,
                }
                for mode in check.MODES
                for split in ("train", "tuning")
                for _ in range(8)
            ]
        )
        excess = check.excess_over_unseen(scores, "train", "tuning")
        loss = check.membership_from_loss(np.ones(8), np.ones(8))
        tails = check.patient_tails(scores)
        complete = check.grid_coverage(scores, [32], [32], ("train", "tuning"), 128, 4)
        self.assertTrue(
            check.comparison_summary(excess, loss, 4, tails, complete).startswith(
                "CHECKS PASSED:"
            )
        )
        missing = check.grid_coverage(
            scores, [32, 128], [32], ("train", "tuning"), 128, 4
        )
        self.assertEqual(
            missing.filter(pl.col("status") == "unsupported_context").height, 6
        )
        self.assertTrue(
            check.comparison_summary(excess, loss, 4, tails, missing).startswith(
                "INCONCLUSIVE:"
            )
        )

    def test_input_validation_rejects_mismatches_before_inference(self):
        vocabulary = {"[PAD]": 0, **{f"CODE_{i}": i for i in range(1, 8)}}
        record = patient(1, [1, 2, 3])
        check.validate_windows({"train": [record]}, vocabulary, model())
        with self.assertRaisesRegex(ValueError, "Missing required"):
            check.validate_windows({"train": [record]}, vocabulary, model(values=True))
        bad = {**record, "age": torch.tensor([1.0])}
        with self.assertRaisesRegex(ValueError, "Misaligned"):
            check.validate_windows({"train": [bad]}, vocabulary, model())
        bad = {**record, "code": torch.tensor([1, 2, 99])}
        with self.assertRaisesRegex(ValueError, "Invalid code"):
            check.validate_windows({"train": [bad]}, vocabulary, model())
        with self.assertRaisesRegex(ValueError, "Vocabulary"):
            check.validate_windows({"train": [record]}, {"[PAD]": 0}, model())

    def test_training_settings_discover_hydra_and_preserve_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hydra").mkdir()
            (root / "version_0").mkdir()
            population = root / "population.csv"
            (root / "hydra/config.yaml").write_text(
                f"paths:\n  population: {population}\ntraining:\n  max_len: 64\n"
                "  cutoff_date: {year: 2015, month: 1, day: 2}\n"
            )
            args = Namespace(
                checkpoint=root / "version_0/best.ckpt",
                run_config=None,
                population=None,
                cutoff_date=None,
                context_length=None,
            )
            check.training_settings(args)
            self.assertEqual(args.population, population)
            self.assertEqual(args.cutoff_date, date(2015, 1, 2))
            self.assertEqual(args.context_length, 64)
            args.population = root / "override.csv"
            args.context_length = 32
            check.training_settings(args)
            self.assertEqual(args.population, root / "override.csv")
            self.assertEqual(args.context_length, 32)

    def test_original_statistics_and_cli_outputs(self):
        auc = check.membership_from_loss(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertEqual(auc["auc"][0], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(root / "model.ckpt", model())
            torch.save(
                {"[PAD]": 0, **{f"CODE_{i}": i for i in range(1, 8)}},
                root / "vocabulary.pt",
            )
            for split, offset in (("train", 0), ("tuning", 10)):
                torch.save(
                    [patient(offset + i, [1, 2, 3, 4, 5]) for i in range(3)],
                    root / f"subject_data_{split}.pt",
                )
            command = [
                sys.executable,
                "-m",
                "bonsai.run.memorization_check",
                "--sampling-design",
                "legacy",
                "--checkpoint",
                str(root / "model.ckpt"),
                "--data-dir",
                str(root),
                "--output-dir",
                str(root / "output"),
                "--device",
                "cpu",
                "--prompt-events",
                "2",
                "--continuation-events",
                "1",
                "2",
                "--min-patients",
                "2",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {p.name for p in (root / "output").iterdir()},
                {
                    "excess.csv",
                    "loss_check.csv",
                    "summary.txt",
                    "patient_tails.csv",
                    "run_manifest.json",
                    "coverage.csv",
                    "patient_flag_rates.csv",
                    "report.txt",
                    "flagged_comparisons.csv",
                    "flagged_loss_checks.csv",
                },
            )
            self.assertIn(
                "PRIMARY: GENERATION WITHOUT PATIENT CONTEXT — NOT RUN",
                (root / "output/report.txt").read_text(),
            )
            self.assertEqual(
                pl.read_csv(root / "output/excess.csv")["X"].unique().to_list(), [2]
            )
            self.assertIn("INCONCLUSIVE", (root / "output/summary.txt").read_text())
            manifest = json.loads((root / "output/run_manifest.json").read_text())
            for field in (
                "setup_seconds",
                "generation_scoring_seconds",
                "loss_seconds",
            ):
                self.assertGreaterEqual(manifest["timing"][field], 0)
            self.assertIn(
                "Estimated 100,000-patient/group run",
                (root / "output/summary.txt").read_text(),
            )
            self.assertEqual(
                manifest["checkpoint_sha256"], check.file_hash(root / "model.ckpt")
            )
            self.assertEqual(manifest["status"], "EXPLORATORY_ONLY")
            self.assertEqual(set(manifest["generation_modes"]), set(check.MODES))
            self.assertEqual(
                set(pl.read_csv(root / "output/excess.csv")["mode"].to_list()),
                set(check.MODES),
            )
            self.assertIsNone(manifest["release_threshold"])
            repeated = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("not overwritten", repeated.stderr)
            self.assertEqual(
                json.loads((root / "output/run_manifest.json").read_text()), manifest
            )
            duplicate_grid = subprocess.run(
                [
                    *command,
                    "--output-dir",
                    str(root / "deduplicated"),
                    "--prompt-events",
                    "2",
                    "2",
                    "--continuation-events",
                    "1",
                    "1",
                    "2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(duplicate_grid.returncode, 0, duplicate_grid.stderr)
            deduplicated = pl.read_csv(root / "deduplicated/excess.csv")
            self.assertTrue((deduplicated["patients_train"] == 3).all())
            self.assertEqual(
                deduplicated.height, pl.read_csv(root / "output/excess.csv").height
            )


if __name__ == "__main__":
    unittest.main()
