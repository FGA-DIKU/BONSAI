"""Reassess completed patient scores with a different overlap-diversity threshold.

No model, source patient files or generation are needed. Original reports are
unchanged. Requires patient_scores.csv with unique_predicted_tokens; aggregate
reports or sampled trajectories alone cannot establish full-population rates.

python -m bonsai.run.reanalyze_memorization --report-dir results \
    --output-dir results_review --min-unique-overlap-tokens 3
"""

import argparse
import json
from pathlib import Path

import polars as pl

from bonsai.run.memorization_check import (
    INSPECTION_HORIZON,
    PARTIAL_REPRODUCTION,
    patient_flag_rates,
    patient_tails,
    write_outputs,
)


def reanalyze(report_dir, output_dir, minimum_unique):
    if minimum_unique < 1:
        raise ValueError("Minimum unique token threshold must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new or empty output directory.")
    scores = pl.read_csv(
        report_dir / "patient_scores.csv",
        schema_overrides={
            name: pl.Float64 for name in ("exact_match", "overlap", "rare_codes")
        },
        null_values=["nan", "NaN"],
    ).with_columns(
        pl.col("exact_match", "overlap", "rare_codes").fill_null(float("nan"))
    )
    if (
        "unique_predicted_tokens" not in scores.columns
        or scores["unique_predicted_tokens"].null_count()
    ):
        raise ValueError(
            "Saved scores lack complete diversity counts. Rerun the cached check with inspection export; sampled examples cannot recover counts for every patient."
        )
    manifest = json.loads((report_dir / "run_manifest.json").read_text())
    manifest["settings"]["min_unique_overlap_tokens"] = minimum_unique
    manifest["settings"]["output_dir"] = str(output_dir)
    manifest["patient_inspection"] = None
    manifest["reanalysis"] = {
        "source_report": str(report_dir.resolve()),
        "source_trajectory_exports": manifest.get("reanalysis", {}).get(
            "source_trajectory_exports", str(report_dir.resolve())
        ),
        "source_run_timing": manifest.pop("timing", None),
        "note": "Only patient-level reproduction flags are reassessed; raw metrics, group comparisons and loss AUC are unchanged. Lower thresholds may flag patients without saved trajectories. Low-diversity exact matches remain flags.",
    }
    applicable = pl.col("Y") >= INSPECTION_HORIZON
    exact = ((pl.col("exact_match") == 1) & applicable).fill_null(False)
    overlap = (
        pl.col("overlap").is_finite()
        & (pl.col("overlap") >= PARTIAL_REPRODUCTION)
        & (pl.col("unique_predicted_tokens") >= minimum_unique)
        & applicable
    ).fill_null(False)
    updated = scores.with_columns(
        (exact | overlap).alias("flagged"),
        pl.concat_str(
            [
                pl.when(exact).then(pl.lit("exact_continuation")).otherwise(pl.lit("")),
                pl.when(overlap)
                .then(pl.lit("overlap_ge_threshold"))
                .otherwise(pl.lit("")),
            ],
            separator=";",
        )
        .str.strip_chars(";")
        .alias("flag_reasons"),
    )
    write_outputs(
        pl.read_csv(report_dir / "excess.csv"),
        pl.read_csv(report_dir / "loss_check.csv"),
        output_dir,
        manifest["settings"]["min_patients"],
        patient_tails(scores, minimum_unique),
        manifest,
        pl.read_csv(report_dir / "coverage.csv"),
        patient_flag_rates(scores, minimum_unique),
    )
    updated.write_csv(output_dir / "patient_scores.csv")
    with (output_dir / "summary.txt").open("a") as stream:
        stream.write(
            f"\nReanalysis of {report_dir.resolve()}. Trajectories remain in the original report directory.\n{manifest['reanalysis']['note']}\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-unique-overlap-tokens", type=int, default=3)
    args = parser.parse_args()
    reanalyze(args.report_dir, args.output_dir, args.min_unique_overlap_tokens)


if __name__ == "__main__":
    main()
