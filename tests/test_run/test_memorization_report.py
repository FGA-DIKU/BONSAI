"""A supplemental pass must not masquerade as a no-context extraction result."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from bonsai.run.report_memorization import plot_excess_grid, report_data, write_report


def saved_report(directory, cohort=None):
    directory.mkdir(parents=True)
    manifest = {
        "evidence_summary": "CHECKS PASSED: supplemental comparisons only.",
        "sampling_design": "fixed",
        "settings": {
            "min_patients": 50,
            "max_patients": 10000,
            "random_fraction": 0.8,
            "rarity_reference_patients": 100000,
            "rarity_calibration_patients": 20000,
        },
    }
    if cohort:
        manifest["analysis_cohort"] = cohort
        manifest["evidence_summary"] = "CHECK: subgroup difference needs review."
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for metric in ("exact_match", "overlap", "rare_codes"):
        rows.append(
            {
                "mode": "code_only",
                "X": 32,
                "Y": 16,
                "score": metric,
                "patients_train": 100,
                "patients_unseen": 100,
                "train": 0.5,
                "unseen": 0.5,
                "excess": -0.02 if cohort else 0.0,
                "lower": -0.03 if cohort else -0.01,
                "upper": -0.01 if cohort else 0.01,
            }
        )
    rows[-1]["patients_train"] = 5  # Rare-score denominator, not cohort size.
    pl.DataFrame(rows).write_csv(directory / "excess.csv")


class TestMemorizationReport(unittest.TestCase):
    def test_primary_subgroup_findings_are_in_the_combined_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "supplemental")
            saved_report(root / "primary")
            saved_report(root / "primary/cohorts/uncommon", "uncommon")
            for name in ("supplemental", "primary"):
                path = root / name / "run_manifest.json"
                metadata = json.loads(path.read_text())
                metadata.update(
                    checkpoint_sha256="same-model", cohort_id_sha256="same-patients"
                )
                if name == "primary":
                    metadata.update(
                        analysis_role="primary_patient_code_only",
                        completion_status="complete",
                        checks_status="CHECK",
                    )
                path.write_text(json.dumps(metadata))
            (root / "primary/primary_summary.txt").write_text(
                "Primary subgroup requires review.\n"
            )
            write_report(
                root / "supplemental",
                root / "combined",
                primary_report=root / "primary",
            )
            flags = pl.read_csv(root / "combined/flagged_comparisons.csv")
            self.assertEqual(flags.height, 2)
            self.assertEqual(set(flags["analysis"]), {"primary/cohorts/uncommon"})
            self.assertEqual(
                set(flags["report_dir"]),
                {str((root / "primary/cohorts/uncommon").resolve())},
            )

    def test_context_zero_and_supplemental_cohort_fingerprints_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "supplemental")
            path = root / "supplemental/run_manifest.json"
            source = json.loads(path.read_text())
            source.update(checkpoint_sha256="same-model", cohort_id_sha256="cohort-a")
            path.write_text(json.dumps(source))
            primary = root / "primary"
            primary.mkdir()
            metadata = {
                "analysis_role": "primary_patient_code_only",
                "completion_status": "complete",
                "checks_status": "CHECKS PASSED",
                "checkpoint_sha256": "same-model",
                "cohort_id_sha256": "cohort-b",
            }
            (primary / "run_manifest.json").write_text(json.dumps(metadata))
            (primary / "primary_summary.txt").write_text(
                "Context-zero per-patient comparisons.\n"
            )
            with self.assertRaisesRegex(ValueError, "patient cohorts differ"):
                write_report(
                    root / "supplemental", root / "rejected", primary_report=primary
                )
            metadata["cohort_id_sha256"] = "cohort-a"
            (primary / "run_manifest.json").write_text(json.dumps(metadata))
            report = write_report(
                root / "supplemental", root / "combined", primary_report=primary
            )
            self.assertIn("Context-zero per-patient", report.read_text())

    def test_subgroup_findings_and_low_denominators_do_not_disappear(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "results")
            saved_report(root / "results/cohorts/uncommon", "uncommon")
            saved_report(root / "results/diversity_filtered/cohorts/random", "random")
            path = write_report(root / "results", root / "report")
            text = path.read_text()
            self.assertIn("WITHOUT PATIENT CONTEXT — NOT RUN", text)
            self.assertIn("cohorts/uncommon", text)
            self.assertIn("diversity_filtered/cohorts/random", text)
            self.assertIn("assessable generation comparisons: 2/3", text)
            self.assertIn(
                "-2.000 percentage points [-3.000, -1.000]; tuning higher", text
            )
            flags = pl.read_csv(root / "report/flagged_comparisons.csv")
            self.assertEqual(flags.height, 4)
            self.assertNotIn("rare_codes", flags["score"].to_list())
            self.assertTrue(flags["needs_review"].all())
            self.assertFalse((root / "results/report.txt").exists())

    def test_no_findings_still_writes_an_empty_searchable_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "results")
            write_report(root / "results")
            flags = pl.read_csv(root / "results/flagged_comparisons.csv")
            self.assertEqual(flags.height, 0)
            self.assertIn("score", flags.columns)
            self.assertIn(
                "CONTEXT-CONDITIONED", (root / "results/report.txt").read_text()
            )

    def test_loss_only_finding_is_visible_without_generation_findings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "results")
            pl.DataFrame(
                [
                    {
                        "patients_train": 100,
                        "patients_unseen": 100,
                        "loss_train": 1.0,
                        "loss_unseen": 2.0,
                        "auc": 0.8,
                        "auc_lower": 0.7,
                        "auc_upper": 0.9,
                    }
                ]
            ).write_csv(root / "results/loss_check.csv")
            write_report(root / "results")
            self.assertIn("Loss CHECK", (root / "results/report.txt").read_text())
            flags = pl.read_csv(root / "results/flagged_loss_checks.csv")
            self.assertEqual(flags.height, 1)
            self.assertEqual(flags["auc"].item(), 0.8)

    def test_primary_and_supplemental_statuses_keep_their_scope_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "supplemental", "random")
            source_path = root / "supplemental/run_manifest.json"
            source = json.loads(source_path.read_text())
            source["checkpoint_sha256"] = "same-checkpoint"
            source["cohort_id_sha256"] = "same-cohort"
            source_path.write_text(json.dumps(source))
            primary = root / "primary"
            primary.mkdir()
            metadata = {
                "analysis_role": "primary_patient_code_only",
                "cohort_id_sha256": "same-cohort",
                "checks_status": "CHECKS PASSED",
                "checkpoint_sha256": "same-checkpoint",
            }
            (primary / "run_manifest.json").write_text(json.dumps(metadata))
            (primary / "primary_summary.txt").write_text(
                "PRIMARY TEST\nCHECKS PASSED for this attack.\n"
            )
            result = write_report(
                root / "supplemental", root / "combined", primary_report=primary
            )
            text = result.read_text()
            self.assertIn(
                "CHECK: primary or supplemental findings require review", text
            )
            self.assertLess(text.index("PRIMARY TEST"), text.index("SUPPLEMENTAL:"))
            self.assertNotIn("NOT RUN", text)
            metadata["completion_status"] = "matching_complete"
            (primary / "run_manifest.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                write_report(
                    root / "supplemental", root / "unfinished", primary_report=primary
                )
            metadata["completion_status"] = "complete"
            metadata["checkpoint_sha256"] = "different-checkpoint"
            (primary / "run_manifest.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "different checkpoints"):
                write_report(
                    root / "supplemental", root / "rejected", primary_report=primary
                )

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib optional")
    def test_grid_handles_underpowered_cells_and_empty_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved_report(root / "results", "random")
            manifest, frame = report_data(root / "results")
            plot_excess_grid([(manifest, frame)], root / "grid.png")
            self.assertGreater((root / "grid.png").stat().st_size, 1000)
            plot_excess_grid([(manifest, frame.head(0))], root / "empty.png")
            self.assertFalse((root / "empty.png").exists())


if __name__ == "__main__":
    unittest.main()
