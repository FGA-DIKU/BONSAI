"""Group findings must be inspectable even below individual-trigger horizons."""

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import polars as pl
import torch
from test_memorization_check import model, patient, save_checkpoint

from bonsai.functional.memorization_cache import (
    CachedWindows,
    FixedCohortWindows,
    save_cache,
)
from bonsai.functional.memorization_inspection import PatientInspection
from bonsai.run import memorization_check as check
from bonsai.run.audit_memorization_cell import (
    audit_cell,
    rare_code_evidence,
    select_examples,
)


class TestCellAudit(unittest.TestCase):
    def test_fixed_cohort_sep_recovery_preserves_ids_batches_and_scored_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = model()
            network = type(base)(
                **{
                    **{
                        k: v
                        for k, v in base.hparams.items()
                        if k != "value_embedding_mode"
                    },
                    "bias": True,
                    "attention_dropout": 0.0,
                }
            )
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                network.pretrain_head.bias[1] = 10
            network.eval()
            save_checkpoint(root / "model.ckpt", network)
            vocabulary = {"[PAD]": 0, "[SEP]": 7, **{f"C{i}": i for i in range(1, 7)}}
            torch.save(vocabulary, root / "vocabulary.pt")
            pools = {}
            for split, offset, positive in (
                ("train", 0, {0, 1, 2, 8}),
                ("tuning", 100, {0, 8}),
            ):
                rows = []
                for i in range(10):
                    codes = [2] * 32 + [7, 1 if i in positive else 3] * (
                        8 if i % 2 else 32
                    )
                    row = patient(offset + i, codes)
                    row["segment"] -= 1
                    row["sampling_cohort"] = "random" if i < 8 else "uncommon"
                    rows.append(row)
                pools[split] = {
                    "rows": rows,
                    "members": [],
                    "random_fraction": 0.8,
                    "excluded_target": 7,
                }
            signature = {"test": "fixed-sep-recovery"}
            save_cache(
                root / "cache.pt",
                signature,
                10,
                pools,
                set(range(1, 7)),
                {1, 3},
                sampling={"test": True},
            )
            datasets = {
                split: FixedCohortWindows(pool) for split, pool in pools.items()
            }
            source = root / "random_report"
            source.mkdir()
            exporter = PatientInspection(source, vocabulary, 1, 32, 0.9)
            horizons = [1, 2, 4, 8, 16, 32, 64]
            check.score_all_patients(
                network,
                datasets,
                [32],
                horizons,
                10,
                set(range(1, 7)),
                {1, 3},
                batch_size=4,
                context=128,
                mode="true_features",
                sep_token_id=7,
                supply_sep=True,
                inspection=exporter,
            )
            exporter.close()
            scores = pl.read_csv(source / "patient_scores.csv")
            scores.filter(pl.col("sampling_cohort") == "random").write_csv(
                source / "patient_scores.csv"
            )
            (source / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(root / "model.ckpt"),
                        "checkpoint_sha256": check.file_hash(root / "model.ckpt"),
                        "vocabulary_sha256": check.file_hash(root / "vocabulary.pt"),
                        "analysis_cohort": "random",
                        "evaluation_cache": {
                            "path": str(root / "cache.pt"),
                            "signature": signature,
                        },
                        "effective_context": 128,
                        "settings": {
                            "data_dir": str(root),
                            "train_split": "train",
                            "heldout_split": "tuning",
                            "continuation_events": horizons,
                            "max_patients": 10,
                            "batch_size": 4,
                            "device": "cpu",
                            "sep_excluded_from_training_loss": True,
                        },
                    }
                )
            )
            args = Namespace(
                report_dir=source,
                cohort=None,
                mode="true_features",
                context=32,
                horizon=16,
                metric="rare_codes",
                top_per_split=0,
                median_per_split=0,
                seed=0,
                subject_id=None,
                all_rare_matches=True,
                evaluation_cache=None,
                checkpoint=None,
                vocabulary=None,
                device="cpu",
                output_dir=root / "audit",
            )
            audit_cell(args)
            recovered = pl.read_csv(args.output_dir / "rare_code_review_cases.csv")
            self.assertEqual(set(recovered["subject_id"]), {0, 1, 2, 100})
            selected = pl.read_csv(args.output_dir / "selected_patients.csv")
            self.assertTrue(selected["replay_matches_saved_scores"].all())
            self.assertEqual(selected["true_rare_occurrences"].to_list(), [8] * 4)
            self.assertEqual(selected["matched_rare_occurrences"].to_list(), [8] * 4)
            rows = [
                json.loads(line)
                for line in (args.output_dir / "unflagged_examples.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                all(
                    row["supplied_sep_positions"] == list(range(0, 16, 2))
                    for row in rows
                )
            )
            self.assertTrue(
                all(row["scored_positions"] == 8 and not row["flagged"] for row in rows)
            )

    def test_new_cache_metadata_uses_full_training_patient_frequencies(self):
        source = [{"code": torch.tensor([1, 1])} for _ in range(1000)] + [
            {"code": torch.tensor([2, 2])}
        ]
        countable, rare, metadata = check.code_groups(
            {"[PAD]": 0, "COMMON": 1, "RARE": 2}, source, return_metadata=True
        )
        self.assertEqual(countable, {1, 2})
        self.assertEqual(rare, {2})
        self.assertEqual(metadata["patient_counts"], {2: 1})
        self.assertEqual(metadata["training_patients"], 1001)

    def test_rare_evidence_counts_occurrences_and_does_not_invent_frequencies(self):
        evidence = rare_code_evidence(
            np.array([1, 1, 3]),
            np.array([1, 3, 3, 4]),
            {1, 3, 4},
            {"A": 1, "B": 3, "C": 4},
        )
        self.assertEqual(evidence["matched_rare_occurrences"], 2)
        self.assertEqual(evidence["true_rare_occurrences"], 3)
        self.assertFalse(evidence["population_frequencies_available"])
        self.assertTrue(
            all(row["training_patient_count"] is None for row in evidence["codes"])
        )
        with_counts = rare_code_evidence(
            [1, 1],
            [1],
            {1},
            {"A": 1},
            {"patient_counts": {1: 3}, "training_patients": 10000},
        )
        self.assertEqual(with_counts["codes"][0]["training_patient_count"], 3)
        self.assertAlmostEqual(
            with_counts["codes"][0]["training_patient_fraction"], 0.0003
        )

    def test_all_rare_matches_replays_every_positive_case_below_minimum_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = model()
            settings = {
                key: value
                for key, value in base.hparams.items()
                if key != "value_embedding_mode"
            }
            network = type(base)(**{**settings, "bias": True, "attention_dropout": 0.0})
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                network.pretrain_head.bias[1] = 10
            network.eval()
            save_checkpoint(root / "model.ckpt", network)
            vocabulary = {"[PAD]": 0, "[SEP]": 7, **{f"C{i}": i for i in range(1, 7)}}
            torch.save(vocabulary, root / "vocabulary.pt")
            pools = {}
            for split, offset, positives in (("train", 0, 3), ("tuning", 100, 1)):
                rows = [
                    patient(
                        offset + i,
                        [2, 2] + ([1, 1] if i < positives else [3, 3]) + [2, 2],
                    )
                    for i in range(8)
                ]
                pools[split] = {
                    "rows": rows,
                    "pools": {4: list(range(8)), "loss": list(range(8))},
                }
            signature = {"test": "all-rare-matches"}
            save_cache(root / "cache.pt", signature, 8, pools, set(range(1, 7)), {1, 3})
            windows = {split: CachedWindows(pool) for split, pool in pools.items()}
            scores = check.score_all_patients(
                network,
                windows,
                [2],
                [2],
                8,
                set(range(1, 7)),
                {1, 3},
                batch_size=4,
                context=12,
                mode="code_only",
            )
            # Reconstruct the ID-level saved scores without relying on exported examples.
            records = []
            for split, dataset in windows.items():
                for row, metrics in zip(
                    dataset.rows, scores.filter(pl.col("split") == split).to_dicts()
                ):
                    records.append(
                        {**metrics, "subject_id": row["subject_id"], "flagged": False}
                    )
            source = root / "source"
            source.mkdir()
            pl.DataFrame(records).write_csv(source / "patient_scores.csv")
            manifest = {
                "checkpoint": str(root / "model.ckpt"),
                "checkpoint_sha256": check.file_hash(root / "model.ckpt"),
                "vocabulary_sha256": check.file_hash(root / "vocabulary.pt"),
                "effective_context": 12,
                "inspection_triggers": {},
                "evaluation_cache": {
                    "path": str(root / "cache.pt"),
                    "signature": signature,
                },
                "settings": {
                    "data_dir": str(root),
                    "train_split": "train",
                    "heldout_split": "tuning",
                    "continuation_events": [2],
                    "prompt_events": [2],
                    "max_patients": 8,
                    "batch_size": 4,
                    "device": "cpu",
                    "min_unique_overlap_tokens": 3,
                },
            }
            (source / "run_manifest.json").write_text(json.dumps(manifest))
            args = Namespace(
                report_dir=source,
                cohort=None,
                mode="code_only",
                context=2,
                horizon=2,
                metric="rare_codes",
                top_per_split=1,
                median_per_split=2,
                seed=0,
                subject_id=None,
                all_rare_matches=True,
                evaluation_cache=None,
                checkpoint=None,
                vocabulary=None,
                device="cpu",
                output_dir=root / "audit",
            )
            audit_cell(args)
            selected = pl.read_csv(root / "audit/selected_patients.csv")
            reviews = selected.filter(pl.col("reason") == "rare_code_match_review")
            self.assertEqual(reviews.height, 4)
            self.assertEqual(
                pl.read_csv(root / "audit/rare_code_review_cases.csv").height, 4
            )
            self.assertEqual(set(reviews["subject_id"]), {0, 1, 2, 100})
            self.assertTrue(selected["replay_matches_saved_scores"].all())
            self.assertEqual(
                selected.filter(pl.col("reason") == "zero_score_control").height, 4
            )
            text = (root / "audit/audit.txt").read_text()
            self.assertIn("RARE-CODE REVIEW", text)
            self.assertIn("RARE-CODE MATCHES", text)
            self.assertIn("Matched 2/2 true rare occurrences", text)
            self.assertIn("Training holders: unavailable", text)
            self.assertTrue(
                pl.read_csv(root / "audit/rare_code_matches.csv")[
                    "training_patient_count"
                ]
                .is_null()
                .all()
            )
            # Newly prepared caches carry actual population counts, without any
            # scan or fabricated frequency based on these 16 evaluated patients.
            save_cache(
                root / "cache_with_counts.pt",
                signature,
                8,
                pools,
                set(range(1, 7)),
                {1, 3},
                rare_metadata={
                    "patient_counts": {1: 3, 3: 5},
                    "training_patients": 10000,
                },
            )
            args.evaluation_cache = root / "cache_with_counts.pt"
            args.output_dir = root / "audit_with_counts"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bonsai.run.audit_memorization_cell",
                    "--report-dir",
                    str(source),
                    "--mode",
                    "code_only",
                    "--context",
                    "2",
                    "--horizon",
                    "2",
                    "--metric",
                    "rare_codes",
                    "--all-rare-matches",
                    "--top-per-split",
                    "1",
                    "--median-per-split",
                    "2",
                    "--device",
                    "cpu",
                    "--evaluation-cache",
                    str(args.evaluation_cache),
                    "--output-dir",
                    str(args.output_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "Training holders: 3/10000", (args.output_dir / "audit.txt").read_text()
            )

    def test_selects_high_and_median_scores_without_individual_flags(self):
        frame = pl.DataFrame(
            [
                {
                    "split": split,
                    "subject_id": f"{split}_{i}",
                    "overlap": score,
                    "flagged": False,
                }
                for split in ("train", "tuning")
                for i, score in enumerate((0.0, 0.2, 0.4, 1.0))
            ]
        )
        selected, stats = select_examples(
            frame, ("train", "tuning"), "overlap", 1, 1, 0
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(not row["flagged"] for row in selected))
        self.assertEqual(
            [
                row["overlap"]
                for row in selected
                if row["selection_reason"] == "top_score"
            ],
            [1.0, 1.0],
        )
        self.assertEqual(stats["train"]["patients"], 4)
        self.assertAlmostEqual(stats["train"]["mean"], 0.4)
        self.assertGreater(selected[0]["signed_mean_contribution"], 0)
        self.assertLess(selected[2]["signed_mean_contribution"], 0)
        self.assertEqual(
            selected, select_examples(frame, ("train", "tuning"), "overlap", 1, 1, 0)[0]
        )

    def test_replays_unsaved_patients_and_checks_agreement_using_cache_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(root / "model.ckpt", model())
            torch.save(
                {"[PAD]": 0, "[SEP]": 7, **{f"C{i}": i for i in range(1, 7)}},
                root / "vocabulary.pt",
            )
            for split, offset in (("train", 0), ("tuning", 100)):
                torch.save(
                    [
                        patient(offset + i, [1, 2, 7, 3, 4, 7, 5, 6, 7, 1, 2, 3])
                        for i in range(8)
                    ],
                    root / f"subject_data_{split}.pt",
                )
            (root / "config.yaml").write_text(
                "paths:\n  population: null\ntraining:\n  cutoff_date: null\n  max_len: 12\n"
            )
            original = root / "original"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bonsai.run.memorization_check",
                    "--checkpoint",
                    str(root / "model.ckpt"),
                    "--data-dir",
                    str(root),
                    "--run-config",
                    str(root / "config.yaml"),
                    "--sampling-design",
                    "legacy",
                    "--max-patients",
                    "8",
                    "--cache-patients",
                    "8",
                    "--min-patients",
                    "2",
                    "--prompt-events",
                    "2",
                    "--continuation-events",
                    "1",
                    "2",
                    "4",
                    "--batch-size",
                    "4",
                    "--evaluation-cache",
                    str(root / "cache.pt"),
                    "--output-dir",
                    str(original),
                    "--device",
                    "cpu",
                    "--export-patient-inspection",
                    "--sep-excluded-from-training-loss",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            existing = [
                json.loads(line)
                for line in (original / "unflagged_examples.jsonl")
                .read_text()
                .splitlines()
            ]
            frame = pl.read_csv(original / "patient_scores.csv")
            ids = []
            for split in ("train", "tuning"):
                saved = {
                    str(row["subject_id"])
                    for row in existing
                    if row["split"] == split
                    and row["mode"] == "code_only"
                    and row["Y"] == 2
                }
                candidates = frame.filter(
                    (pl.col("split") == split)
                    & (pl.col("mode") == "code_only")
                    & (pl.col("Y") == 2)
                )
                ids.append(
                    next(
                        str(i) for i in candidates["subject_id"] if str(i) not in saved
                    )
                )
                (
                    root / f"subject_data_{split}.pt"
                ).unlink()  # Replay must not load population files.
            command = [
                sys.executable,
                "-m",
                "bonsai.run.audit_memorization_cell",
                "--report-dir",
                str(original),
                "--mode",
                "code_only",
                "--context",
                "2",
                "--horizon",
                "2",
                "--metric",
                "overlap",
                "--subject-id",
                *ids,
                "--device",
                "cpu",
            ]
            audited = subprocess.run(
                [*command, "--output-dir", str(root / "audit")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            metadata = json.loads((root / "audit/run_manifest.json").read_text())[
                "cell_audit"
            ]
            self.assertEqual(metadata["selected_examples"], 2)
            self.assertEqual(metadata["replay_mismatches"], 0)
            self.assertEqual(metadata["original_batches_replayed"], 2)
            self.assertEqual(
                pl.read_csv(root / "audit/cell_patient_scores.csv").height, 16
            )
            audit = (root / "audit/audit.txt").read_text()
            self.assertIn("GROUP-LEVEL CELL AUDIT", audit)
            self.assertIn("Selection: requested_id", audit)
            for identifier in ids:
                self.assertIn(f"Patient {identifier}", audit)
            # A differing replay must remain explicitly labelled, never passed
            # off as the lost original trajectory.
            frame.with_columns(
                pl.when((pl.col("mode") == "code_only") & (pl.col("Y") == 2))
                .then(pl.lit(0.123))
                .otherwise(pl.col("overlap"))
                .alias("overlap")
            ).write_csv(original / "patient_scores.csv")
            drift = subprocess.run(
                [*command, "--output-dir", str(root / "drift")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(drift.returncode, 0, drift.stderr)
            self.assertIn("REPLAY DISAGREEMENT", (root / "drift/audit.txt").read_text())


if __name__ == "__main__":
    unittest.main()
