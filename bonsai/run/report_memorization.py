"""Build a readable report and figures from completed memorization analyses.

python -m bonsai.run.report_memorization --report-dir RESULTS

Reads aggregate CSV/JSON reports; no checkpoint, patient trajectories or
inference. --include-patient-distributions additionally reads saved split/loss
columns (no IDs). Figures require pip install '.[plots]'. Use --no-plots for text only.
Existing context-conditioned checks are supplemental. They do not substitute
for a no-patient-context extraction experiment.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

METRICS = ("exact_match", "overlap", "rare_codes")


def completed_reports(root):
    """Visit known analysis folders without picking up individual audit replays."""
    for view in (Path(), Path("diversity_filtered")):
        for cohort in (Path(), Path("cohorts/random"), Path("cohorts/uncommon")):
            directory = root / view / cohort
            if (directory / "excess.csv").is_file():
                yield directory


def report_data(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    # Empty filtered tables have only CSV headers, so inferred types are strings.
    excess = pl.read_csv(
        directory / "excess.csv",
        schema_overrides={
            **{
                key: pl.Float64
                for key in ("train", "unseen", "excess", "lower", "upper")
            },
            **{
                key: pl.Int64 for key in ("X", "Y", "patients_train", "patients_unseen")
            },
        },
    )
    minimum = manifest["settings"]["min_patients"]
    ready = (
        pl.col("lower").is_finite()
        & pl.col("upper").is_finite()
        & (pl.min_horizontal("patients_train", "patients_unseen") >= minimum)
    ).fill_null(False)
    excess = excess.with_columns(ready.alias("assessable")).with_columns(
        (pl.col("assessable") & ((pl.col("lower") > 0) | (pl.col("upper") < 0))).alias(
            "needs_review"
        )
    )
    return manifest, excess


def plot_excess_grid(reports, output):
    """Compare cohorts/modes on one common percentage-point scale per metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    cells = sorted(
        {tuple(row) for _, frame in reports for row in frame.select("X", "Y").rows()}
    )
    if not cells:
        return
    rows = [
        (manifest.get("analysis_cohort", "combined"), mode)
        for manifest, frame in reports
        for mode in sorted(frame["mode"].unique())
    ]
    if not rows:
        return
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(12, len(cells) * 0.65), max(9, len(rows) * 1.1)),
        constrained_layout=True,
    )
    for axis, metric in zip(axes, METRICS):
        values = np.full((len(rows), len(cells)), np.nan)
        flags, weak = [], []
        row_index = 0
        for _manifest, frame in reports:
            for mode in sorted(frame["mode"].unique()):
                part = frame.filter(
                    (pl.col("mode") == mode) & (pl.col("score") == metric)
                )
                for record in part.iter_rows(named=True):
                    column = cells.index((record["X"], record["Y"]))
                    if record["assessable"] and math.isfinite(record["excess"]):
                        values[row_index, column] = 100 * record["excess"]
                        if record["needs_review"]:
                            flags.append((column, row_index))
                    else:
                        weak.append((column, row_index))
                row_index += 1
        limit = max(
            0.1, float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.1
        )
        colors = plt.get_cmap("RdBu_r").copy()
        colors.set_bad("#dddddd")
        plotted = axis.imshow(
            values,
            aspect="auto",
            cmap=colors,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        )
        if flags:
            axis.scatter(
                *zip(*flags), marker="o", facecolors="none", edgecolors="black", s=55
            )
        if weak:
            axis.scatter(*zip(*weak), marker="x", color="#666666", s=20)
        axis.set_yticks(
            range(len(rows)), [f"{cohort} / {mode}" for cohort, mode in rows]
        )
        axis.set_xticks(range(len(cells)), [f"{x}/{y}" for x, y in cells], rotation=60)
        axis.set_title(metric + " — training minus tuning, percentage points")
        axis.set_xlabel("Context / horizon (sequence positions)")
        fig.colorbar(plotted, ax=axis, label="Percentage points")
    fig.suptitle(
        "SUPPLEMENTAL context-conditioned comparisons\n"
        "Circle: pointwise 95% CI excludes zero; grey/x: absent or insufficient evidence"
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_primary(directory, output):
    from bonsai.run.plot_memorization import plot_reports

    for source in completed_reports(directory):
        plot_reports(source, output / source.relative_to(directory))


def primary_findings(directory):
    rows, schema = [], None
    for source in completed_reports(directory):
        _, frame = report_data(source)
        schema = {"analysis": pl.String, "report_dir": pl.String, **frame.schema}
        for record in frame.filter(pl.col("needs_review")).iter_rows(named=True):
            rows.append(
                {
                    "analysis": "primary/" + str(source.relative_to(directory)),
                    "report_dir": str(source.resolve()),
                    **record,
                }
            )
    return rows, schema


def write_report(
    report_dir,
    output_dir=None,
    plots=False,
    include_patient_distributions=False,
    primary_report=None,
):
    """Write report.txt and an uncapped index of group comparisons needing review."""
    report_dir = Path(report_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else report_dir
    directories = list(completed_reports(report_dir))
    root_manifest_path = report_dir / "run_manifest.json"
    if (
        root_manifest_path.is_file()
        and json.loads(root_manifest_path.read_text()).get("analysis_role")
        == "primary_patient_code_only"
    ):
        primary_report = primary_report or report_dir
        directories = []  # These are primary context-zero cells, not supplemental ones.
    primary_lines = []
    if primary_report is not None:
        primary_report = Path(primary_report).resolve()
        primary = json.loads((primary_report / "run_manifest.json").read_text())
        if primary.get("analysis_role") != "primary_patient_code_only":
            raise ValueError(
                "--primary-report must be a completed context-zero per-patient report."
            )
        if primary.get("completion_status", "complete") != "complete":
            raise ValueError(
                "Primary context-zero run did not complete; inspect its server log before reporting a result."
            )
        primary_lines = [
            "PRIMARY REPORT: " + str(primary_report),
            *(primary_report / "primary_summary.txt").read_text().splitlines(),
            "",
        ]
    if not directories and not primary_lines:
        raise ValueError(
            "No completed memorization reports found (excess.csv required)."
        )
    data = [(directory, *report_data(directory)) for directory in directories]
    if (
        primary_lines
        and data
        and primary["checkpoint_sha256"] != data[0][1]["checkpoint_sha256"]
    ):
        raise ValueError(
            "Primary and supplemental reports concern different checkpoints."
        )
    if primary_lines and data:
        first = data[0][1]
        same = (
            primary.get("cohort_id_sha256") == first.get("cohort_id_sha256")
            if first.get("cohort_id_sha256")
            else primary["settings"]["max_patients"]
            == first["settings"]["max_patients"]
            and primary["evaluation_cache"]["signature"]
            == first["evaluation_cache"]["signature"]
        )
        if not same:
            raise ValueError(
                "Primary and supplemental patient cohorts differ. Use the same fixed cache and --max-patients in both checks."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if primary_lines and not data:
        if plots:
            plot_primary(primary_report, output_dir / "plots/primary")
        findings, schema = primary_findings(primary_report)
        if schema is not None:
            pl.DataFrame(findings, schema=schema).write_csv(
                output_dir / "flagged_comparisons.csv"
            )
        (output_dir / "report.txt").write_text(
            "\n".join(
                [
                    "BONSAI MEMORIZATION REPORT",
                    *primary_lines,
                    "SUPPLEMENTAL ANALYSES: not included in this report.",
                    "Primary subgroup findings are indexed in flagged_comparisons.csv; each row identifies its source folder.",
                    f"Figures: {output_dir / 'plots/primary'}"
                    if plots
                    else "Figures were not generated by this report command.",
                ]
            )
            + "\n"
        )
        return output_dir / "report.txt"
    lines = [
        "BONSAI MEMORIZATION REPORT",
        "",
        "PRIMARY: GENERATION WITHOUT PATIENT CONTEXT — NOT RUN BY THESE ANALYSES",
        "The existing files evaluate prompted greedy continuations and teacher-forced loss.",
        "They cannot establish the result of an unconditional sampling experiment.",
        "",
        "SUPPLEMENTAL: CONTEXT-CONDITIONED REPRODUCTION",
        "Statuses below apply only to the supplemental checks. A pass is not sharing approval.",
        "",
    ]
    if primary_lines:
        lines[2:6] = primary_lines
        statuses = [
            primary["checks_status"],
            *[
                metadata.get(
                    "checks_status",
                    metadata.get("evidence_summary", "INCONCLUSIVE").split(":", 1)[0],
                )
                for _, metadata, _ in data
            ],
        ]
        if "CHECK" in statuses:
            overall = "CHECK: primary or supplemental findings require review; inspect the respective sections."
        elif any(status != "CHECKS PASSED" for status in statuses):
            overall = (
                "INCONCLUSIVE: at least one included analysis lacks adequate evidence."
            )
        else:
            overall = "CHECKS PASSED: the primary attack and included supplemental comparisons completed without review triggers."
        lines[1:2] = ["", overall, ""]
    flagged, loss_findings = [], []
    if primary_lines and primary.get("analysis_role") == "primary_patient_code_only":
        flagged, _ = primary_findings(primary_report)
    for directory, manifest, frame in data:
        label = str(directory.relative_to(report_dir)) or "."
        findings = frame.filter(pl.col("needs_review")).sort("mode", "X", "Y", "score")
        status = manifest.get(
            "evidence_summary", "Status unavailable; inspect summary.txt."
        )
        lines.extend([f"[{label}] {status}", f"  Report folder: {directory}"])
        settings = manifest["settings"]
        lines.append(
            f"  Requested patients per split: {settings.get('max_patients', 'not recorded')}; "
            f"assessable generation comparisons: {frame['assessable'].sum()}/{frame.height}; "
            f"comparisons needing review: {findings.height}."
        )
        if (directory / "loss_check.csv").is_file():
            loss = pl.read_csv(directory / "loss_check.csv").row(0, named=True)
            finite = all(
                loss.get(key) is not None and math.isfinite(loss[key])
                for key in ("auc", "auc_lower", "auc_upper")
            )
            enough = (
                min(loss["patients_train"], loss["patients_unseen"])
                >= settings["min_patients"]
            )
            loss_status = "INCONCLUSIVE"
            if finite and enough:
                loss_status = (
                    "CHECK"
                    if not loss["auc_lower"] <= 0.5 <= loss["auc_upper"]
                    else "ASSESSABLE"
                )
            lines.append(
                f"  Loss {loss_status}: patients train/tuning {loss['patients_train']}/{loss['patients_unseen']}; "
                f"AUC {loss['auc']:.4f} [{loss['auc_lower']:.4f}, {loss['auc_upper']:.4f}]."
            )
            if loss_status == "CHECK":
                loss_findings.append(
                    {"analysis": label, "report_dir": str(directory), **loss}
                )
        if filters := manifest.get("diversity_filters") or (
            [manifest["diversity_filter"]] if manifest.get("diversity_filter") else []
        ):
            lines.append(
                "  Diversity subset: "
                + "; ".join(
                    f"at least {f['minimum_unique_tokens']} distinct {f['basis']} clinical codes"
                    for f in filters
                )
                + ". Loss/AUC remains unfiltered."
            )
        for record in findings.iter_rows(named=True):
            flagged.append({"analysis": label, "report_dir": str(directory), **record})
            direction = "training higher" if record["excess"] > 0 else "tuning higher"
            lines.append(
                f"  CHECK {record['mode']} X={record['X']} Y={record['Y']} {record['score']}: "
                f"{100 * record['excess']:+.3f} percentage points "
                f"[{100 * record['lower']:+.3f}, {100 * record['upper']:+.3f}]; {direction}; "
                f"n={record['patients_train']}/{record['patients_unseen']}."
            )
        lines.append("")
    first_manifest = data[0][1]
    settings = first_manifest["settings"]
    if fingerprint := first_manifest.get("checkpoint_sha256"):
        lines.extend([f"Checkpoint SHA-256: {fingerprint}", ""])
    lines.extend(
        [
            "HOW TO INTERPRET THE SCORES",
            "exact_match: fraction of patients whose entire scored continuation matches in order.",
            "overlap: mean per-patient fraction of true clinical occurrences reproduced, ignoring order.",
            "rare_codes: the same fraction restricted to rare codes; only patients with true rare occurrences contribute.",
            "Each generated occurrence can match at most one true occurrence. SEP and other special codes do not count for overlap.",
            "Rare-code metric: held by fewer than 0.1% of full filtered training-window patients; this differs from uncommon cohort selection.",
            "Excess = training mean minus tuning mean. All displayed differences and intervals above are in percentage points.",
            "AUC is a ranking signal from teacher-forced loss, not the percentage of patients memorized.",
            "Intervals are pointwise, not corrected across comparisons; excluding zero needs investigation, not a memorization diagnosis.",
            "Tuning may have influenced checkpoint selection. Passing these checks does not establish equivalence or absence of memorization.",
            "",
            "SAMPLING",
        ]
    )
    if first_manifest.get("sampling_design") == "fixed":
        lines.extend(
            [
                f"Fixed random/uncommon IDs across contexts and modes; random fraction {settings.get('random_fraction', 0.8):.0%}.",
                "Random patients are selected without using model scores. Additional uncommon patients use code-pair frequency surprisal, not BONSAI predictions.",
                f"Requested rarity fitting/calibration patients: {settings.get('rarity_reference_patients', 'not recorded')}/{settings.get('rarity_calibration_patients', 'not recorded')}, reserved independently from evaluated training IDs.",
                "Shared calibration cutoffs and matched rarity/history-availability strata are used in both splits.",
                "Uncommon and combined cohorts are enriched stress samples; the random cohort supplies the population-facing comparison.",
                "Short histories contribute only where eligible; actual metric denominators above can be smaller than the requested cohort.",
                "Inspect sampling_report.json and sampling_balance.csv for cutoffs, actual allocations and shortages.",
            ]
        )
        if (report_dir / "sampling_report.json").is_file():
            sampling = json.loads((report_dir / "sampling_report.json").read_text())
            lines.append(
                f"Actual reserved fitting/calibration counts: {sampling.get('reference_patients_reserved', 'not recorded')}/{sampling.get('calibration_patients_reserved', 'not recorded')}."
            )
            for split, counts in sampling.get("splits", {}).items():
                lines.append(
                    f"  {split}: prepared random={counts['random']}, uncommon={counts['uncommon']}/{counts['uncommon_requested']}; complete={counts['complete']}. Prepared cache capacity may exceed the evaluated subset."
                )
    else:
        lines.append(
            "Legacy sampling: separate eligible samples by context/loss; consult the saved manifest."
        )
    lines.extend(
        [
            "",
            "WHERE TO INVESTIGATE",
            "flagged_comparisons.csv: every assessable generation comparison whose interval excludes zero, with source folders.",
            "flagged_loss_checks.csv: every assessable loss-AUC comparison whose interval excludes 0.5.",
            "Each source folder: summary.txt, excess.csv, coverage.csv, patient_flag_rates.csv and patient_tails.csv.",
            "Patient flags and group differences are distinct: a flagged mean difference need not produce an individual trajectory flag.",
            "For all rare-code-positive patients behind a finding (both splits), run:",
            "  python -m bonsai.run.audit_memorization_cell --report-dir SOURCE_FOLDER --mode MODE --context X --horizon Y --metric rare_codes --all-rare-matches --output-dir NEW_AUDIT_FOLDER --device cuda",
            "Then open NEW_AUDIT_FOLDER/audit.txt in vim. Requires the original cache/checkpoint/vocabulary; replays are checked against saved scores.",
            "Existing audit.txt files contain individual exact/overlap flags and sampled examples, not every contributor to a group difference.",
            "Low diversity: inspect patient_flag_rates.csv; diversity_filtered/ reports require post-run reanalysis and must be compared with the unfiltered results.",
            "",
        ]
    )
    if plots:
        from bonsai.run.plot_memorization import plot_reports

        if primary_lines:
            plot_primary(primary_report, output_dir / "plots/primary")

        for directory, _manifest, _frame in data:
            destination = output_dir / "plots" / directory.relative_to(report_dir)
            plot_reports(directory, destination, include_patient_distributions)
        for view in (Path(), Path("diversity_filtered")):
            group = [
                (m, f)
                for d, m, f in data
                if d.parent == report_dir / view / "cohorts" or d == report_dir / view
            ]
            if group:
                destination = output_dir / "plots" / view / "cohort_excess.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                plot_excess_grid(group, destination)
        lines.append(
            f"FIGURES: {output_dir / 'plots'}"
            + (
                " (primary in primary/; supplemental results in their respective folders)."
                if primary_lines
                else " (supplemental checks)."
            )
        )
        lines.extend(
            [
                "  *_metrics.png: train/tuning mean scores and excess confidence intervals over horizons.",
                "  cohort_excess.png: compare modes and cohorts; circles mark pointwise intervals excluding zero.",
                "  coverage.png and *_denominators.png: eligibility and actual counts contributing to each metric.",
                "  *_score_tails.png: median, 95th percentile and maximum reproduction; descriptive, not additional tests.",
                "  *_patient_rates.png and *_exact_diversity.png: reproduction rates and exact-match diversity counts.",
                "  *_diversity_rates.png: overall low-diversity and individual-review rates, where saved counts support them.",
                "  diversity_retention.png: fraction retained at each horizon, when a filtered analysis exists.",
                "  loss_auc.png: existing teacher-forced membership ranking comparison.",
            ]
        )
        if include_patient_distributions:
            lines.append(
                "  loss_distributions.png: saved split/loss distributions where patient_losses.csv exists; no individual IDs read."
            )
    else:
        lines.append(
            "Figures not generated. Run python -m bonsai.run.report_memorization --report-dir RESULTS after installing '.[plots]'."
        )
    schema = {"analysis": pl.String, "report_dir": pl.String, **data[0][2].schema}
    pl.DataFrame(flagged, schema=schema).write_csv(
        output_dir / "flagged_comparisons.csv"
    )
    loss_schema = {
        "analysis": pl.String,
        "report_dir": pl.String,
        "patients_train": pl.Int64,
        "patients_unseen": pl.Int64,
        **{
            key: pl.Float64
            for key in ("loss_train", "loss_unseen", "auc", "auc_lower", "auc_upper")
        },
    }
    pl.DataFrame(loss_findings, schema=loss_schema).write_csv(
        output_dir / "flagged_loss_checks.csv"
    )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n")
    return output_dir / "report.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--primary-report",
        type=Path,
        help="Include a completed context-zero per-patient report before supplemental results",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--include-patient-distributions",
        action="store_true",
        help="Also plot saved per-patient losses, without reading patient IDs or trajectories",
    )
    args = parser.parse_args()
    print(
        write_report(
            args.report_dir,
            args.output_dir,
            plots=not args.no_plots,
            include_patient_distributions=args.include_patient_distributions,
            primary_report=args.primary_report,
        )
    )


if __name__ == "__main__":
    main()
