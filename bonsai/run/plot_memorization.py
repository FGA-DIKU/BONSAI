"""Plot completed aggregate reports without loading patient data or a model.

Install plotting support with: pip install '.[plots]'
Run: python -m bonsai.run.plot_memorization --report-dir memorization_full
PNG figures go in report-dir/plots (or --output-dir). No inference is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def plot_reports(report_dir, output_dir, include_patient_distributions=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    excess = pl.read_csv(report_dir / "excess.csv")
    tails = pl.read_csv(report_dir / "patient_tails.csv")
    loss = (
        pl.read_csv(report_dir / "loss_check.csv").row(0, named=True)
        if (report_dir / "loss_check.csv").is_file()
        else None
    )
    manifest = json.loads((report_dir / "run_manifest.json").read_text())
    train = manifest["settings"]["train_split"]
    heldout = manifest["settings"]["heldout_split"]
    minimum_unique = manifest.get("inspection_triggers", {}).get(
        "minimum_unique_predicted_clinical_tokens_for_overlap", 3
    )
    conditional = manifest.get("separator_policy", {}).get(
        "true_features_supplies_sep", False
    )
    scope = (
        "Primary / "
        if manifest.get("analysis_role") == "primary_patient_code_only"
        else "Supplemental / "
    ) + manifest.get("analysis_cohort", "combined")
    if manifest.get("diversity_filter"):
        subset = manifest["diversity_filter"]
        scope += f" / ≥{subset['minimum_unique_tokens']} unique {subset['basis']} codes"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ("exact_match", "overlap", "rare_codes")
    for mode in sorted(excess["mode"].unique()):
        part = excess.filter(pl.col("mode") == mode)
        if part.is_empty():
            continue
        contexts = sorted(part["X"].unique())
        colors = dict(
            zip(contexts, plt.cm.viridis(np.linspace(0.1, 0.9, len(contexts))))
        )
        title = mode + (
            " (supplied SEP; conditional)"
            if conditional and mode == "true_features"
            else ""
        )
        if manifest.get("analysis_role") == "primary_patient_code_only":
            title += " (probability sampling; public DOB seed)"
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for column, metric in enumerate(metrics):
            for context in contexts:
                cell = part.filter(
                    (pl.col("score") == metric) & (pl.col("X") == context)
                ).sort("Y")
                if cell.is_empty():
                    continue
                x = cell["Y"].to_numpy()
                for group, style in (("train", "-"), ("unseen", "--")):
                    label = train if group == "train" else heldout
                    axes[0, column].plot(
                        x,
                        cell[group].to_numpy(),
                        style,
                        marker="o",
                        color=colors[context],
                        label=f"X={context}, {label}",
                    )
                axes[1, column].plot(
                    x,
                    cell["excess"].to_numpy(),
                    marker="o",
                    color=colors[context],
                    label=f"X={context}",
                )
                axes[1, column].fill_between(
                    x,
                    cell["lower"].to_numpy(),
                    cell["upper"].to_numpy(),
                    color=colors[context],
                    alpha=0.15,
                )
            axes[0, column].set_title(metric)
            axes[0, column].set_ylim(-0.02, 1.02)
            axes[0, column].set_ylabel("Mean patient score")
            axes[1, column].set_ylabel("Train − comparison (pointwise 95% CI)")
            axes[1, column].axhline(0, color="grey", linewidth=1)
            for axis in axes[:, column]:
                axis.set_xscale("log", base=2)
                axis.set_xlabel("Continuation positions (Y)")
                axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        fig.suptitle(
            scope + " / " + title + " — missing/undersampled cells: see coverage.csv"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_metrics.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        for axis, metric, numerator, label in (
            (axes[0, 0], "exact_match", "patients_scoring_1", "Exact continuation (%)"),
            (
                axes[0, 1],
                "overlap",
                "patients_high_overlap_diverse"
                if "patients_high_overlap_diverse" in tails.columns
                else "patients_scoring_ge_90pct",
                f"At least 90% overlap, ≥{minimum_unique} unique codes (%)"
                if "patients_high_overlap_diverse" in tails.columns
                else "At least 90% overlap (%)",
            ),
        ):
            for context in contexts:
                for split, style in ((train, "-"), (heldout, "--")):
                    cell = tails.filter(
                        (pl.col("mode") == mode)
                        & (pl.col("score") == metric)
                        & (pl.col("X") == context)
                        & (pl.col("split") == split)
                    ).sort("Y")
                    if cell.is_empty():
                        continue
                    axis.plot(
                        cell["Y"].to_numpy(),
                        100 * cell[numerator].to_numpy() / cell["patients"].to_numpy(),
                        style,
                        marker="o",
                        color=colors[context],
                        label=f"X={context}, {split}",
                    )
            axis.set(xlabel="Continuation positions (Y)", ylabel=label, ylim=(-2, 102))
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            scope + " / " + title + " — proportions per cell, not percentages memorized"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_patient_rates.png", dpi=180)
        plt.close(fig)

    if loss is not None:
        fig, axis = plt.subplots(figsize=(7, 3))
        axis.hlines(0, loss["auc_lower"], loss["auc_upper"], linewidth=3)
        axis.plot(loss["auc"], 0, "o")
        axis.axvline(0.5, color="grey", linestyle="--")
        axis.set(
            xlim=(0, 1),
            yticks=[],
            xlabel="Loss membership AUC (95% approximate CI)",
            title=scope + "\nLower loss ranks training membership; chance = 0.5",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "loss_auc.png", dpi=180)
        plt.close(fig)

    from bonsai.run.memorization_diagnostic_plots import plot_diagnostics

    plot_diagnostics(report_dir, output_dir, include_patient_distributions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--include-patient-distributions",
        action="store_true",
        help="Also plot saved per-patient loss distributions, reading split/loss columns only; no inference",
    )
    args = parser.parse_args()
    plot_reports(
        args.report_dir,
        args.output_dir or args.report_dir / "plots",
        args.include_patient_distributions,
    )


if __name__ == "__main__":
    main()
