"""Reanalysed cohort folders must find the original local patient exports."""

import json
import tempfile
import unittest
from pathlib import Path

from bonsai.run.inspect_memorization import examples


class TestInspectionProvenance(unittest.TestCase):
    def test_nested_old_report_links_keep_cohort_and_all_diversity_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, review, cohort = [
                root / name for name in ("original", "review", "cohort")
            ]
            for path in (original, review, cohort):
                path.mkdir()
            rows = [
                {
                    "subject_id": identifier,
                    "split": "train",
                    "mode": "code_only",
                    "X": 32,
                    "Y": 16,
                    "sampling_cohort": group,
                    "scores": {
                        "unique_reference_tokens": ref,
                        "unique_predicted_tokens": pred,
                    },
                }
                for identifier, group, ref, pred in (
                    (1, "random", 4, 1),
                    (2, "uncommon", 4, 4),
                    (3, "random", 2, 4),
                    (4, "random", 4, 4),
                )
            ]
            (original / "flagged_trajectories.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (review / "run_manifest.json").write_text(
                json.dumps({"reanalysis": {"source_trajectory_exports": str(original)}})
            )
            (cohort / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "analysis_cohort": "random",
                        "diversity_filters": [
                            {"basis": "reference", "minimum_unique_tokens": 3},
                            {"basis": "predicted", "minimum_unique_tokens": 2},
                        ],
                        "reanalysis": {"source_trajectory_exports": str(review)},
                    }
                )
            )
            self.assertEqual([row["subject_id"] for row in examples(cohort, None)], [4])
            (original / "run_manifest.json").write_text(
                json.dumps({"reanalysis": {"source_trajectory_exports": str(cohort)}})
            )
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                list(examples(cohort, None))


if __name__ == "__main__":
    unittest.main()
