"""Diagnostic figures handle sparse cells and keep patient-loss reads optional."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from bonsai.run.memorization_diagnostic_plots import plot_diagnostics


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib optional")
class TestDiagnosticPlots(unittest.TestCase):
    def test_sparse_cells_diversity_counts_and_optional_loss_distributions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "settings": {
                            "train_split": "train",
                            "heldout_split": "tuning",
                            "min_patients": 50,
                        },
                        "diversity_filter": {
                            "basis": "reference",
                            "minimum_unique_tokens": 3,
                        },
                    }
                )
            )
            tails, coverage, rates = [], [], []
            for split in ("train", "tuning"):
                for horizon, n in ((1, 100), (32, 98), (64, 2)):
                    cell = {"mode": "code_only", "split": split, "X": 32, "Y": horizon}
                    coverage.append(
                        {
                            **cell,
                            "patients": n,
                            "status": "not_applicable_diversity"
                            if horizon == 1
                            else "complete"
                            if n >= 50
                            else "too_few_patients",
                        }
                    )
                    # Rare denominators differ, and one split has no rare-score cell.
                    for metric in ("exact_match", "overlap", "rare_codes"):
                        if metric == "rare_codes" and split == "tuning":
                            continue
                        tails.append(
                            {
                                **cell,
                                "score": metric,
                                "patients": n if metric != "rare_codes" else 1,
                                "median": 0.0,
                                "p95": 0.8,
                                "max": 1.0,
                            }
                        )
                    rates.append(
                        {
                            **cell,
                            "patients": n,
                            "inspection_applicable": horizon >= 32,
                            "patients_exact_flagged": min(4, n),
                            "patients_flagged_low_diversity": min(2, n),
                            "patients_diversity_unknown": 0,
                            "percent_low_diversity": 100 * min(2, n) / n,
                            "percent_flagged": 100 * min(4, n) / n
                            if horizon >= 32
                            else None,
                        }
                    )
            pl.DataFrame(tails).write_csv(root / "patient_tails.csv")
            pl.DataFrame(coverage).write_csv(root / "coverage.csv")
            pl.DataFrame(rates).write_csv(root / "patient_flag_rates.csv")
            pl.DataFrame(
                [
                    {
                        "mode": "code_only",
                        "split": "train",
                        "X": 32,
                        "Y": 32,
                        "patients_before": 100,
                        "patients_retained": 40,
                        "patients_excluded": 60,
                    }
                ]
            ).write_csv(root / "diversity_exclusions.csv")
            (root / "patient_losses.csv").write_text(
                "unreadable sentinel; optional file must stay unopened"
            )
            plot_diagnostics(root, root / "aggregate")
            expected = {
                "code_only_denominators.png",
                "code_only_score_tails.png",
                "coverage.png",
                "code_only_exact_diversity.png",
                "diversity_retention.png",
                "code_only_diversity_rates.png",
            }
            self.assertEqual(
                {path.name for path in (root / "aggregate").glob("*.png")}, expected
            )
            pl.DataFrame(
                {
                    "subject_id": [
                        "unused-id-a",
                        "unused-id-b",
                        "unused-id-c",
                        "unused-id-d",
                    ],
                    "split": ["train", "train", "tuning", "tuning"],
                    "mean_next_code_loss": [0.5, 1.0, 1.0, 2.0],
                }
            ).write_csv(root / "patient_losses.csv")
            plot_diagnostics(
                root, root / "with_losses", include_patient_distributions=True
            )
            self.assertTrue((root / "with_losses/loss_distributions.png").is_file())


if __name__ == "__main__":
    unittest.main()
