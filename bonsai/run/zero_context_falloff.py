"""Every-step descriptive curves from stored context-zero draws; no inference.

python -m bonsai.run.zero_context_falloff --report-dir CONTEXT_ZERO \
    --output-dir NEW_FALLOFF_FOLDER --plots

Fixed-cohort curves retain patients through the maximum horizon. All-available
curves also use shorter stored continuations, with denominators at every step.
"""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch

from bonsai.functional.memorization_cache import cohort_identity, load_fixed_references
from bonsai.functional.memorization_metrics import difference_from_summary
from bonsai.run.memorization_check import file_hash
from bonsai.run.zero_context_check import reference_continuations

METRICS = ("exact_match", "overlap", "rare_codes", "position_accuracy")
COHORTS = ("combined", "random", "uncommon")
VIEWS = ("fixed", "all_available")


def prefix_scores(predicted, truth, countable, rare):
    """All prefixes in one linear scan, with occurrence-limited unordered overlap."""
    scores = np.full((len(truth), len(METRICS)), np.nan)
    predicted_counts, true_counts = Counter(), Counter()
    hits = rare_hits = total = rare_total = 0
    exact = True
    for step, (written, actual) in enumerate(zip(predicted, truth)):
        predicted_counts[written] += 1
        if written in countable and predicted_counts[written] <= true_counts[written]:
            hits += 1
            rare_hits += written in rare
        true_counts[actual] += 1
        if actual in countable:
            total += 1
            rare_total += actual in rare
            if true_counts[actual] <= predicted_counts[actual]:
                hits += 1
                rare_hits += actual in rare
        exact = exact and written == actual
        scores[step] = (
            float(exact),
            hits / total if total else np.nan,
            rare_hits / rare_total if rare_total else np.nan,
            float(written == actual),
        )
    return scores


class Moments:
    """Online per-step counts/means/sample variances; memory independent of N."""

    def __init__(self, horizon):
        self.n = np.zeros((horizon, len(METRICS)), dtype=np.int64)
        self.mean = np.zeros_like(self.n, dtype=np.float64)
        self.m2 = np.zeros_like(self.mean)

    def update(self, scores):
        n, mean, m2 = (
            self.n[: len(scores)],
            self.mean[: len(scores)],
            self.m2[: len(scores)],
        )
        valid = np.isfinite(scores)
        n[valid] += 1
        delta = scores[valid] - mean[valid]
        mean[valid] += delta / n[valid]
        m2[valid] += delta * (scores[valid] - mean[valid])


def summarize(groups, splits, horizon, minimum):
    rows = []
    for view in VIEWS:
        for cohort in COHORTS:
            train, tuning = [groups[view, cohort, split] for split in splits]
            for step in range(horizon):
                for metric, name in enumerate(METRICS):
                    i = step, metric
                    n1, n2 = int(train.n[i]), int(tuning.n[i])
                    p1, p2 = (
                        train.mean[i] if n1 else np.nan,
                        tuning.mean[i] if n2 else np.nan,
                    )
                    delta, lower, upper = p1 - p2, np.nan, np.nan
                    if min(n1, n2) >= 2:
                        delta, lower, upper = difference_from_summary(
                            p1,
                            train.m2[i] / (n1 - 1),
                            n1,
                            p2,
                            tuning.m2[i] / (n2 - 1),
                            n2,
                            name in ("exact_match", "position_accuracy"),
                        )
                    rows.append(
                        {
                            "view": view,
                            "cohort": cohort,
                            "Y": step + 1,
                            "score": name,
                            "patients_train": n1,
                            "patients_unseen": n2,
                            "train": p1,
                            "unseen": p2,
                            "excess": delta,
                            "lower": lower,
                            "upper": upper,
                            "adequate_n": min(n1, n2) >= minimum,
                        }
                    )
    return pl.DataFrame(rows)


def write_report(frame, directory, horizon):
    lines = [
        "CONTEXT-ZERO EVERY-STEP FALLOFF (DESCRIPTIVE)",
        f"Horizon: 1..{horizon}; original saved draws, no new inference.",
        "Fixed: same patients have reference AND stored predictions through the maximum horizon.",
        "All available: eligible patients change with Y; changes can reflect cohort composition.",
        "Exact match is cumulative prefix equality, including SEP. Position accuracy scores only token Y.",
        "Overlap/rare scores use occurrence-limited clinical-code recall; rare denominators exclude prefixes without true rare codes.",
        "Even in the fixed cohort, rare-score eligibility changes when a patient's first rare code appears.",
        "Raw unfiltered scores; no diversity exclusions. Raw sequence positions include separators; public seed excluded.",
        "Pointwise 95% intervals, not simultaneous bands or additional pass/fail tests. Original check statuses are unchanged.",
        "Scores are percentages; excess and interval endpoints are percentage points. n is metric-specific.",
        "Values with adequate_n=false are under the source run's minimum sample count; see every_step.csv.",
    ]
    for view in VIEWS:
        for cohort in COHORTS:
            lines += [
                "",
                f"{view.upper()} / {cohort}",
                "   Y metric                n_train n_tuning   train%  tuning%  excess_pp       95% interval_pp",
            ]
            part = frame.filter((pl.col("view") == view) & (pl.col("cohort") == cohort))
            for row in part.iter_rows(named=True):
                lines.append(
                    f"{row['Y']:4d} {row['score']:21s} {row['patients_train']:7d} {row['patients_unseen']:8d} "
                    f"{100 * row['train']:8.3f} {100 * row['unseen']:8.3f} {100 * row['excess']:+10.3f} "
                    f"[{100 * row['lower']:+9.3f}, {100 * row['upper']:+9.3f}]"
                )
    (directory / "report.txt").write_text("\n".join(lines) + "\n")


def plot_curves(frame, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cohort in COHORTS:
        fig, axes = plt.subplots(3, len(METRICS), figsize=(18, 10), squeeze=False)
        for column, metric in enumerate(METRICS):
            for view, style in (("fixed", "-"), ("all_available", "--")):
                part = frame.filter(
                    (pl.col("cohort") == cohort)
                    & (pl.col("view") == view)
                    & (pl.col("score") == metric)
                ).sort("Y")
                x = part["Y"].to_numpy()
                for split, color in (("train", "tab:blue"), ("unseen", "tab:orange")):
                    axes[0, column].plot(
                        x,
                        100 * part[split].to_numpy(),
                        style,
                        color=color,
                        label=f"{split} / {view}",
                    )
                    axes[2, column].plot(
                        x, part[f"patients_{split}"].to_numpy(), style, color=color
                    )
                axes[1, column].plot(
                    x, 100 * part["excess"].to_numpy(), style, label=view
                )
                if view == "fixed":
                    axes[1, column].fill_between(
                        x,
                        100 * part["lower"].to_numpy(),
                        100 * part["upper"].to_numpy(),
                        alpha=0.15,
                    )
            axes[0, column].set_title(metric)
            axes[0, column].set_ylim(-2, 102)
            axes[1, column].axhline(0, color="grey", linewidth=1)
            for axis in axes[:, column]:
                axis.set_xlabel("Generated position / prefix length Y")
                axis.grid(alpha=0.2)
        axes[0, 0].set_ylabel("Score (%)")
        axes[1, 0].set_ylabel("Train − tuning (percentage points)")
        axes[2, 0].set_ylabel("Evaluated patients (metric-specific)")
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].legend(fontsize=7)
        fig.suptitle(
            f"Context zero / {cohort}: every-step descriptive curves; fixed-cohort pointwise 95% bands"
        )
        fig.tight_layout()
        fig.savefig(directory / f"falloff_{cohort}.png", dpi=160)
        plt.close(fig)


def run(args):
    if args.plots and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("Install '.[plots]' or omit --plots for table/text only.")
    manifest = json.loads((args.report_dir / "run_manifest.json").read_text())
    if (
        manifest.get("analysis_role") != "primary_patient_code_only"
        or manifest.get("completion_status") != "complete"
        or manifest.get("analysis_cohort")
        or manifest.get("diversity_filter")
    ):
        raise ValueError("Use the completed, unfiltered combined context-zero report.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory.")
    generation = manifest["generation"]
    for path, expected in (
        (Path(generation["path"]), generation["sha256"]),
        (Path(manifest["vocabulary"]), manifest["vocabulary_sha256"]),
        (Path(manifest["generation_assignment"]), generation["assignment_sha256"]),
    ):
        if file_hash(path) != expected:
            raise ValueError(f"Source artifact changed: {path}")
    records, countable, rare, cache = load_fixed_references(
        manifest, manifest["settings"]["max_patients"], args.evaluation_cache
    )
    if cohort_identity(records) != manifest["cohort_id_sha256"]:
        raise ValueError("Cached patient cohort changed.")
    vocabulary = torch.load(
        manifest["vocabulary"], map_location="cpu", weights_only=False
    )
    rows, assignments = reference_continuations(
        records, vocabulary[generation["start_token"]], generation["start_token"]
    )
    saved = pl.read_csv(
        manifest["generation_assignment"], schema_overrides={"subject_id": pl.String}
    )
    if saved.to_dicts() != assignments:
        raise ValueError("Patient-to-draw assignments changed.")
    samples = np.load(generation["path"], mmap_mode="r")
    available = generation["generated_positions"]
    if len(samples) != len(rows) or len(available) != len(rows):
        raise ValueError("Stored prediction rows do not match the patient assignments.")
    horizon = args.max_horizon if args.max_horizon is not None else samples.shape[1]
    if not 1 <= horizon <= samples.shape[1]:
        raise ValueError("Maximum horizon must fit the stored prediction matrix.")
    splits = manifest["settings"]["train_split"], manifest["settings"]["heldout_split"]
    groups = {
        (view, cohort, split): Moments(horizon)
        for view in VIEWS
        for cohort in COHORTS
        for split in splits
    }
    for index, (row, generated) in enumerate(zip(rows, available)):
        length = min(horizon, len(row["code"]), generated)
        if generated < 0 or generated > samples.shape[1]:
            raise ValueError("Invalid generated-length metadata.")
        if not length:
            continue
        written = samples[index, :length]
        if (written < 0).any():
            raise ValueError("Missing predictions inside a stored continuation.")
        scores = prefix_scores(written, row["code"][:length].numpy(), countable, rare)
        for cohort in ("combined", row["sampling_cohort"]):
            groups["all_available", cohort, row["split"]].update(scores)
            if length == horizon:
                groups["fixed", cohort, row["split"]].update(scores)
        if (index + 1) % 10000 == 0:
            print(
                f"Every-step scoring: {index + 1:,}/{len(rows):,} patients", flush=True
            )
    frame = summarize(groups, splits, horizon, manifest["settings"]["min_patients"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_csv(args.output_dir / "every_step.csv")
    write_report(frame, args.output_dir, horizon)
    if args.plots:
        plot_curves(frame, args.output_dir)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_report": str(args.report_dir.resolve()),
                "generation_sha256": generation["sha256"],
                "cohort_id_sha256": manifest["cohort_id_sha256"],
                "evaluation_cache": cache,
                "max_horizon": horizon,
                "analysis_role": "descriptive_only",
                "completion_status": "complete",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Every-step table and report: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path)
    parser.add_argument("--max-horizon", type=int)
    parser.add_argument("--plots", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
