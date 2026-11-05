"""Server-local patient inspection exports; these files contain patient data."""

import csv
import json
import math
from collections import defaultdict

import numpy as np


class PatientInspection:
    """Stream all flags and retain a uniform bounded sample of unflagged cells.

    A patient may occur in multiple cells. Flags describe reproduction, not a
    diagnosis of memorization. History is the evaluated window, not necessarily
    the entire original patient record (training truncation/cache limits apply).
    """

    def __init__(
        self,
        directory,
        vocabulary,
        examples,
        minimum_horizon,
        overlap_threshold,
        minimum_unique_overlap=3,
        stream_unflagged=False,
    ):
        self.directory = directory
        self.tokens = {identifier: token for token, identifier in vocabulary.items()}
        self.examples = examples
        self.minimum_horizon = minimum_horizon
        self.overlap_threshold = overlap_threshold
        self.minimum_unique_overlap = minimum_unique_overlap
        self.unflagged_stream = (
            (directory / "unflagged_examples.partial.jsonl").open("x")
            if stream_unflagged
            else None
        )
        self.rng = np.random.default_rng(0)
        self.seen = defaultdict(int)
        self.samples = defaultdict(list)
        self.score_file = (directory / "patient_scores.csv").open("x", newline="")
        self.writer = csv.DictWriter(
            self.score_file,
            fieldnames=[
                "subject_id",
                "split",
                "mode",
                "sampling_cohort",
                "X",
                "Y",
                "scored_positions",
                "exact_match",
                "overlap",
                "rare_codes",
                "unique_predicted_tokens",
                "unique_reference_tokens",
                "flagged",
                "flag_reasons",
            ],
        )
        self.writer.writeheader()
        self.flag_file = (directory / "flagged_trajectories.partial.jsonl").open("x")

    def record(self, window, written, split, mode, x, y, scores, supplied_sep, sep_id):
        reasons = []
        if y >= self.minimum_horizon:
            if scores["exact_match"] == 1:
                reasons.append("exact_continuation")
            if (
                scores["overlap"] >= self.overlap_threshold
                and scores["unique_predicted_tokens"] >= self.minimum_unique_overlap
            ):
                reasons.append("overlap_ge_threshold")
        truth = window["code"][x : x + y].tolist()
        supplied = [
            i for i, code in enumerate(truth) if supplied_sep and code == sep_id
        ]
        self.writer.writerow(
            {
                "subject_id": window["subject_id"],
                "split": split,
                "mode": mode,
                "sampling_cohort": window.get("sampling_cohort", "all"),
                "X": x,
                "Y": y,
                "scored_positions": y - len(supplied),
                **scores,
                "flagged": bool(reasons),
                "flag_reasons": ";".join(reasons),
            }
        )
        key = (split, mode, x, y)
        slot = None
        if not reasons and self.unflagged_stream is None:
            self.seen[key] += 1
            if len(self.samples[key]) < self.examples:
                slot = len(self.samples[key])
            elif self.examples:
                candidate = int(self.rng.integers(self.seen[key]))
                if candidate < self.examples:
                    slot = candidate
            if slot is None:
                return

        def sequence(codes):
            return {"ids": codes, "tokens": [self.tokens[code] for code in codes]}

        def features(start, end):
            result = {}
            for field in ("code", "age", "abspos", "segment", "numeric_value"):
                if field in window:
                    result[field] = [
                        None if isinstance(v, float) and not math.isfinite(v) else v
                        for v in window[field][start:end].tolist()
                    ]
            return result

        row = {
            "subject_id": window["subject_id"],
            "split": split,
            "mode": mode,
            "sampling_cohort": window.get("sampling_cohort", "all"),
            "X": x,
            "Y": y,
            "flagged": bool(reasons),
            "flag_reasons": reasons,
            "scores": {k: v if math.isfinite(v) else None for k, v in scores.items()},
            "scope": "evaluated window only; training truncation/cache limits apply",
            "prompt": sequence(window["code"][:x].tolist()),
            "prompt_features": features(0, x),
            "true_continuation": sequence(truth),
            "true_continuation_features": features(x, x + y),
            "predicted_continuation": sequence(written[:y].tolist()),
            "supplied_sep_positions": supplied,
            "later_available_reference": sequence(window["code"][x + y :].tolist()),
            "scored_positions": y - len(supplied),
        }
        if "audit_selection" in window:
            row["audit_selection"] = window["audit_selection"]
        if "rare_code_evidence" in window:
            row["rare_code_evidence"] = window["rare_code_evidence"]
        if "public_seed" in window:
            row["public_seed"] = window["public_seed"]
            row["reference_offset"] = window["reference_offset"]
            row["reference_history"] = sequence(window["reference_history"].tolist())
            row["generation_row"] = window["generation_row"]
        if reasons:
            self.flag_file.write(json.dumps(row, allow_nan=False) + "\n")
        elif self.unflagged_stream is not None:
            self.unflagged_stream.write(json.dumps(row, allow_nan=False) + "\n")
        elif slot == len(self.samples[key]):
            self.samples[key].append(row)
        else:
            self.samples[key][slot] = row

    def close(self):
        self.score_file.close()
        self.flag_file.close()
        (self.directory / "flagged_trajectories.partial.jsonl").rename(
            self.directory / "flagged_trajectories.jsonl"
        )
        if self.unflagged_stream is not None:
            self.unflagged_stream.close()
            (self.directory / "unflagged_examples.partial.jsonl").rename(
                self.directory / "unflagged_examples.jsonl"
            )
        else:
            with (self.directory / "unflagged_examples.jsonl").open("x") as stream:
                for key in sorted(self.samples):
                    for row in self.samples[key]:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")


def write_patient_losses(directory, splits, patients, losses):
    """Loss is a membership ranking signal; do not label individual losses as flags."""
    with (directory / "patient_losses.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["split", "subject_id", "sampling_cohort", "mean_next_code_loss"]
        )
        for split, records, values in zip(splits, patients, losses):
            writer.writerows(
                (
                    split,
                    row["subject_id"],
                    row.get("sampling_cohort", "all"),
                    float(value),
                )
                for row, value in zip(records, values)
            )
