"""Reassess completed patient scores with a different overlap-diversity threshold.

No model, source patient files or generation are needed. Original reports are
unchanged. Requires patient_scores.csv with unique_predicted_tokens; aggregate
reports or sampled trajectories alone cannot establish full-population rates.

python -m bonsai.run.reanalyze_memorization --report-dir results \
    --output-dir results_review --min-unique-overlap-tokens 3

Add --min-diversity-tokens 3 --diversity-basis reference to also write a
diversity_filtered/ analysis with reproduction metrics, excess, counts and
intervals recomputed on retained continuations. Loss/AUC is shared/unfiltered.
Use --diversity-basis predicted only for descriptive sensitivity analysis.
"""

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

from bonsai.functional.memorization_metrics import (
    INSPECTION_HORIZON,
    PARTIAL_REPRODUCTION,
    excess_over_unseen,
    grid_coverage,
    patient_flag_rates,
    patient_tails,
)
from bonsai.run.memorization_check import write_cohort_reports, write_outputs


def reanalyze(
    report_dir,
    output_dir,
    minimum_unique,
    minimum_diversity=None,
    diversity_basis="reference",
):
    if minimum_unique < 1:
        raise ValueError("Minimum unique token threshold must be positive.")
    if minimum_diversity is not None and minimum_diversity < 1:
        raise ValueError("Minimum diversity must be positive.")
    if diversity_basis not in ("reference", "predicted"):
        raise ValueError("Diversity basis must be reference or predicted.")
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
    diversity_column = f"unique_{diversity_basis}_tokens"
    if minimum_diversity is not None and (
        diversity_column not in scores.columns or scores[diversity_column].null_count()
    ):
        raise ValueError(
            f"Saved scores lack {diversity_column}; rerun with the new inference code to record it for every patient. No filtered estimates were written."
        )
    manifest = json.loads((report_dir / "run_manifest.json").read_text())
    fixed = manifest.get("sampling_design") == "fixed" and not manifest.get(
        "analysis_cohort"
    )
    if fixed and not (report_dir / "patient_losses.csv").is_file():
        raise ValueError(
            "Fixed-cohort reanalysis needs patient_losses.csv to recompute cohort AUCs; aggregate AUC cannot be split after the fact."
        )
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

    def cohort_reports(destination, data, metadata):
        if not fixed:
            return
        shutil.copyfile(
            report_dir / "patient_losses.csv", destination / "patient_losses.csv"
        )
        settings = metadata["settings"]
        options = argparse.Namespace(**settings)
        options.output_dir = destination
        options.export_patient_inspection = True
        losses_frame = pl.read_csv(destination / "patient_losses.csv")
        records = [
            losses_frame.filter(pl.col("split") == split).to_dicts()
            for split in (settings["train_split"], settings["heldout_split"])
        ]
        import numpy as np

        losses = [
            np.array([row["mean_next_code_loss"] for row in part]) for part in records
        ]
        write_cohort_reports(
            options, data, records, losses, metadata, metadata["effective_context"]
        )

    cohort_reports(output_dir, updated, manifest)
    with (output_dir / "summary.txt").open("a") as stream:
        stream.write(
            f"\nReanalysis of {report_dir.resolve()}. Trajectories remain in the original report directory.\n{manifest['reanalysis']['note']}\n"
        )
    if minimum_diversity is not None:
        filtered = updated.filter(pl.col(diversity_column) >= minimum_diversity)
        subset_manifest = json.loads(json.dumps(manifest))
        subset_manifest["diversity_filter"] = {
            "basis": diversity_basis,
            "minimum_unique_tokens": minimum_diversity,
            "scope": "generation continuation, per patient/cell; loss AUC remains the unfiltered comparison",
        }
        prior_filters = manifest.get("diversity_filters") or (
            [manifest["diversity_filter"]] if manifest.get("diversity_filter") else []
        )
        subset_manifest["diversity_filters"] = [
            *prior_filters,
            subset_manifest["diversity_filter"],
        ]
        subset_dir = output_dir / "diversity_filtered"
        subset_manifest["settings"]["output_dir"] = str(subset_dir)
        settings = manifest["settings"]
        splits = (settings["train_split"], settings["heldout_split"])
        original_coverage = pl.read_csv(report_dir / "coverage.csv")
        coverage = grid_coverage(
            filtered,
            sorted(original_coverage["X"].unique()),
            sorted(original_coverage["Y"].unique()),
            splits,
            manifest["effective_context"],
            settings["min_patients"],
            minimum_diversity=max(
                f["minimum_unique_tokens"] for f in subset_manifest["diversity_filters"]
            ),
        )
        write_outputs(
            excess_over_unseen(filtered, *splits),
            pl.read_csv(report_dir / "loss_check.csv"),
            subset_dir,
            settings["min_patients"],
            patient_tails(filtered, minimum_unique),
            subset_manifest,
            coverage,
            patient_flag_rates(filtered, minimum_unique),
        )
        filtered.write_csv(subset_dir / "patient_scores.csv")
        cohort_reports(subset_dir, filtered, subset_manifest)
        denominators = (
            updated.group_by("mode", "split", "X", "Y")
            .agg(
                pl.len().alias("patients_before"),
                (pl.col(diversity_column) >= minimum_diversity)
                .sum()
                .alias("patients_retained"),
            )
            .with_columns(
                (pl.col("patients_before") - pl.col("patients_retained")).alias(
                    "patients_excluded"
                )
            )
        )
        denominators.sort("mode", "split", "X", "Y").write_csv(
            subset_dir / "diversity_exclusions.csv"
        )
        with (subset_dir / "summary.txt").open("a") as stream:
            stream.write(
                f"\nSECONDARY SUBSET ANALYSIS: at least {minimum_diversity} distinct {diversity_basis} clinical codes in the continuation. Metrics and excess recomputed on retained patients; loss/AUC reused without filtering. See diversity_exclusions.csv for every denominator.\n"
            )
            if diversity_basis == "predicted":
                stream.write(
                    "Filtering on predictions conditions on model behaviour; use as descriptive sensitivity analysis, not the primary comparison.\n"
                )

    from bonsai.run.report_memorization import write_report

    write_report(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-unique-overlap-tokens", type=int, default=3)
    parser.add_argument(
        "--min-diversity-tokens",
        type=int,
        help="Also recompute generation metrics/excess without low-diversity continuations; leaves unfiltered reports intact",
    )
    parser.add_argument(
        "--diversity-basis", choices=("reference", "predicted"), default="reference"
    )
    args = parser.parse_args()
    reanalyze(
        args.report_dir,
        args.output_dir,
        args.min_unique_overlap_tokens,
        args.min_diversity_tokens,
        args.diversity_basis,
    )


if __name__ == "__main__":
    main()
