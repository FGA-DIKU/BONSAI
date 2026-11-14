"""The context-zero cell must preserve the existing patients, metrics and targets."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch
from test_memorization_check import model, patient, save_checkpoint

from bonsai.functional.memorization_cache import cohort_identity, save_cache
from bonsai.run import memorization_check as check
from bonsai.run.zero_context_check import reference_continuations


class TestZeroContext(unittest.TestCase):
    def test_reference_anchor_is_not_assumed_to_be_first(self):
        records = {"train": [{**patient(1, [5, 6, 1, 2]), "sampling_cohort": "random"}]}
        rows, assignment = reference_continuations(records, 6, "DOB")
        self.assertEqual(rows[0]["code"].tolist(), [1, 2])
        self.assertEqual(assignment[0]["reference_offset"], 2)
        self.assertEqual(rows[0]["reference_history"].tolist(), [5, 6])
        for invalid in ([1, 2], [6, 1, 6]):
            records["train"][0]["code"] = torch.tensor(invalid)
            with self.assertRaisesRegex(ValueError, "No patients were replaced"):
                reference_continuations(records, 6, "DOB")

    def test_same_fixed_ids_original_scores_reuse_and_stored_patient_audit(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                network.pretrain_head.bias[1] = 20
            save_checkpoint(root / "model.ckpt", network)
            vocabulary = {
                "[PAD]": 0,
                **{f"C{i}": i for i in range(1, 6)},
                "DOB": 6,
                "[SEP]": 7,
            }
            torch.save(vocabulary, root / "vocabulary.pt")
            pools, records = {}, {}
            for split, offset in (("train", 0), ("tuning", 100)):
                part = []
                for i in range(10):
                    codes = ([5, 6] if i % 2 == 0 else [6]) + [
                        1 if split == "train" else 2
                    ] * (64 if i % 2 == 0 else 16)
                    part.append(
                        {
                            **patient(offset + i, codes),
                            "sampling_cohort": "random" if i < 8 else "uncommon",
                        }
                    )
                records[split] = part
                pools[split] = {
                    "rows": part,
                    "members": [],
                    "random_fraction": 0.8,
                    "excluded_target": 7,
                }
            signature = {"test": "same-patients"}
            save_cache(
                root / "cache.pt",
                signature,
                10,
                pools,
                set(range(1, 7)),
                {1, 2},
                sampling={"test": True},
            )
            source = root / "source"
            source.mkdir()
            metadata = {
                "checkpoint": str(root / "model.ckpt"),
                "checkpoint_sha256": check.file_hash(root / "model.ckpt"),
                "vocabulary_sha256": check.file_hash(root / "vocabulary.pt"),
                "effective_context": 128,
                "evaluation_cache": {
                    "path": str(root / "cache.pt"),
                    "signature": signature,
                },
                "sampling_complete": True,
                "cohort_id_sha256": cohort_identity(records),
                "settings": {
                    "data_dir": str(root),
                    "max_patients": 10,
                    "min_patients": 2,
                    "train_split": "train",
                    "heldout_split": "tuning",
                    "run_config": "toy",
                    "sep_excluded_from_training_loss": True,
                },
            }
            (source / "run_manifest.json").write_text(json.dumps(metadata))
            command = [
                sys.executable,
                "-m",
                "bonsai.run.zero_context_check",
                "--report-dir",
                str(source),
                "--device",
                "cpu",
                "--batch-size",
                "4",
                "--horizons",
                "16",
                "32",
                "64",
            ]
            result = subprocess.run(
                [*command, "--output-dir", str(root / "zero")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            scores = pl.read_csv(root / "zero/patient_scores.csv")
            self.assertEqual(
                set(scores["subject_id"]), set(range(10)) | set(range(100, 110))
            )
            self.assertEqual(set(scores["X"]), {0})
            self.assertEqual(scores.filter(pl.col("Y") == 16).height, 20)
            self.assertEqual(scores.filter(pl.col("Y") == 64).height, 10)
            self.assertTrue(
                (scores.filter(pl.col("split") == "train")["exact_match"] == 1).all()
            )
            self.assertTrue(
                (scores.filter(pl.col("split") == "tuning")["exact_match"] == 0).all()
            )
            primary = json.loads((root / "zero/run_manifest.json").read_text())
            self.assertEqual(primary["cohort_id_sha256"], metadata["cohort_id_sha256"])
            self.assertEqual(primary["analysis_role"], "primary_patient_code_only")
            samples = np.load(root / "zero/generations.npy")
            assignments = pl.read_csv(root / "zero/generation_assignment.csv")
            for row in assignments.iter_rows(named=True):
                if row["reference_length"] == 16:
                    self.assertTrue((samples[row["row_index"], 16:] == -1).all())
            self.assertIn(
                "Public model input: DOB", (root / "zero/audit.txt").read_text()
            )
            self.assertIn(
                "REFERENCE HISTORY — NEVER MODEL INPUT",
                (root / "zero/audit.txt").read_text(),
            )
            excess = pl.read_csv(root / "zero/excess.csv")
            for metric in check.SCORES:
                self.assertTrue(
                    (excess.filter(pl.col("score") == metric)["excess"] == 1).all()
                )
            reused = subprocess.run(
                [
                    *command,
                    "--reuse-generations",
                    str(root / "zero"),
                    "--checkpoint",
                    str(root / "missing.ckpt"),
                    "--output-dir",
                    str(root / "review"),
                    "--min-diversity-tokens",
                    "3",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertTrue(excess.equals(pl.read_csv(root / "review/excess.csv")))
            self.assertEqual(
                pl.read_csv(
                    root / "review/diversity_filtered/patient_scores.csv"
                ).height,
                0,
            )
            excluded = pl.read_csv(
                root / "review/diversity_filtered/diversity_exclusions.csv"
            )
            self.assertEqual(excluded["patients_before"].sum(), scores.height)
            self.assertEqual(excluded["patients_retained"].sum(), 0)
            self.assertIn(
                "Cohort statuses:", (root / "zero/primary_summary.txt").read_text()
            )
            self.assertIn("random:", (root / "zero/primary_summary.txt").read_text())
            falloff = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bonsai.run.zero_context_falloff",
                    "--report-dir",
                    str(root / "zero"),
                    "--output-dir",
                    str(root / "zero/falloff"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(falloff.returncode, 0, falloff.stderr)
            dense = pl.read_csv(root / "zero/falloff/every_step.csv")
            self.assertEqual(set(dense["Y"]), set(range(1, 65)))
            combined = dense.filter(
                (pl.col("cohort") == "combined") & (pl.col("score") == "exact_match")
            )
            fixed = combined.filter(pl.col("view") == "fixed")
            self.assertEqual(set(fixed["patients_train"]), {5})
            available = combined.filter(pl.col("view") == "all_available")
            self.assertEqual(
                available.filter(pl.col("Y") == 1)["patients_train"].item(), 10
            )
            self.assertEqual(
                available.filter(pl.col("Y") == 17)["patients_train"].item(), 5
            )
            self.assertTrue((fixed["excess"] == 1).all())
            self.assertTrue(excess.equals(pl.read_csv(root / "zero/excess.csv")))
            from bonsai.run.report_memorization import write_report

            overview = write_report(root / "zero", root / "overview")
            self.assertIn("every_step.csv", overview.read_text())
            audited = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bonsai.run.zero_context_check",
                    "--audit-only",
                    str(root / "zero"),
                    "--subject-id",
                    "0",
                    "--output-dir",
                    str(root / "patient_audit"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            self.assertIn("Patient 0", (root / "patient_audit/audit.txt").read_text())
            rare = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bonsai.run.audit_memorization_cell",
                    "--report-dir",
                    str(root / "zero"),
                    "--context",
                    "0",
                    "--horizon",
                    "16",
                    "--mode",
                    "code_only",
                    "--metric",
                    "rare_codes",
                    "--all-rare-matches",
                    "--output-dir",
                    str(root / "rare_audit"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(rare.returncode, 0, rare.stderr)
            self.assertEqual(
                pl.read_csv(root / "rare_audit/rare_code_review_cases.csv").height, 10
            )


if __name__ == "__main__":
    unittest.main()
