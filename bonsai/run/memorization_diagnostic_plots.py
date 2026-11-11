"""Additional descriptive figures from saved reports; no model or generation."""

import json

import numpy as np
import polars as pl


def plot_diagnostics(report_dir, output_dir, include_patient_distributions=False):
    """Expose denominator loss, concentrated scores and repetitive exact matches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((report_dir / "run_manifest.json").read_text())
    scope = (
        "Primary / "
        if manifest.get("analysis_role") == "primary_patient_code_only"
        else "Supplemental / "
    ) + manifest.get("analysis_cohort", "combined")
    if subset := manifest.get("diversity_filter"):
        scope += f" / ≥{subset['minimum_unique_tokens']} unique {subset['basis']} codes"
    tails = pl.read_csv(
        report_dir / "patient_tails.csv",
        schema_overrides={
            **{name: pl.Float64 for name in ("median", "p95", "max")},
            **{name: pl.Int64 for name in ("X", "Y", "patients")},
        },
    )
    splits = (
        manifest["settings"]["train_split"],
        manifest["settings"]["heldout_split"],
    )
    for mode in sorted(tails["mode"].unique()):
        part = tails.filter(pl.col("mode") == mode)
        contexts = sorted(part["X"].unique())
        colors = dict(
            zip(contexts, plt.cm.viridis(np.linspace(0.1, 0.9, len(contexts))))
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for axis, metric in zip(axes, ("exact_match", "overlap", "rare_codes")):
            for context in contexts:
                for split, style in zip(splits, ("-", "--")):
                    cell = part.filter(
                        (pl.col("score") == metric)
                        & (pl.col("X") == context)
                        & (pl.col("split") == split)
                    ).sort("Y")
                    if not cell.is_empty():
                        axis.plot(
                            cell["Y"],
                            cell["patients"],
                            style,
                            marker="o",
                            color=colors[context],
                            label=f"X={context}, {split}",
                        )
            axis.axhline(
                manifest["settings"]["min_patients"],
                color="grey",
                linestyle=":",
                label="Minimum for comparison",
            )
            axis.set(
                xlabel="Horizon (sequence positions)",
                ylabel="Scored patients",
                title=metric,
            )
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(
            scope
            + f" / {mode} — actual metric denominators; absent cells: see coverage.csv"
        )
        fig.savefig(output_dir / f"{mode}_denominators.png", dpi=170)
        plt.close(fig)

        # Maxima can reveal isolated cases, but are descriptive and sample-size dependent.
        fig, axes = plt.subplots(
            2,
            len(contexts),
            figsize=(5 * len(contexts), 7),
            squeeze=False,
            constrained_layout=True,
        )
        for column, context in enumerate(contexts):
            for row, metric in enumerate(("overlap", "rare_codes")):
                axis = axes[row, column]
                for split, color in zip(splits, ("#4c72b0", "#dd8452")):
                    cell = part.filter(
                        (pl.col("score") == metric)
                        & (pl.col("X") == context)
                        & (pl.col("split") == split)
                    ).sort("Y")
                    for statistic, style in (
                        ("median", "-"),
                        ("p95", "--"),
                        ("max", ":"),
                    ):
                        if not cell.is_empty():
                            axis.plot(
                                cell["Y"],
                                cell[statistic],
                                style,
                                color=color,
                                label=f"{split}, {statistic}",
                            )
                axis.set(
                    title=f"{metric}, X={context}",
                    xlabel="Horizon (sequence positions)",
                    ylabel="Patient score",
                    ylim=(-0.02, 1.02),
                )
                axis.set_xscale("log", base=2)
                axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            scope
            + f" / {mode} — score tails (descriptive; maxima depend on sample size)"
        )
        fig.savefig(output_dir / f"{mode}_score_tails.png", dpi=170)
        plt.close(fig)

    if (report_dir / "coverage.csv").is_file():
        coverage = pl.read_csv(report_dir / "coverage.csv")
        if not coverage.is_empty():
            cells = sorted(set(coverage.select("X", "Y").rows()))
            groups = sorted(set(coverage.select("mode", "split").rows()))
            counts = np.zeros((len(groups), len(cells)))
            incomplete, inapplicable = [], []
            for record in coverage.iter_rows(named=True):
                row = groups.index((record["mode"], record["split"]))
                column = cells.index((record["X"], record["Y"]))
                counts[row, column] = record["patients"]
                if record["status"] == "not_applicable_diversity":
                    inapplicable.append((column, row))
                elif record["status"] != "complete":
                    incomplete.append((column, row))
            fig, axis = plt.subplots(
                figsize=(max(12, len(cells) * 0.6), max(4, len(groups) * 0.65)),
                constrained_layout=True,
            )
            plotted = axis.imshow(counts, cmap="Blues", aspect="auto")
            if incomplete:
                axis.scatter(*zip(*incomplete), color="#b2182b", marker="x", s=55)
            if inapplicable:
                axis.scatter(*zip(*inapplicable), color="grey", marker="s", s=35)
            axis.set_xticks(
                range(len(cells)), [f"{x}/{y}" for x, y in cells], rotation=60
            )
            axis.set_yticks(
                range(len(groups)), [f"{mode} / {split}" for mode, split in groups]
            )
            axis.set(
                xlabel="Context / horizon",
                title=scope
                + " — generation coverage\nRed x: incomplete/unsupported; grey square: structurally inapplicable diversity filter",
            )
            fig.colorbar(
                plotted,
                ax=axis,
                label="Evaluated patients (metric denominators may be lower)",
            )
            fig.savefig(output_dir / "coverage.png", dpi=170)
            plt.close(fig)

    if (report_dir / "patient_flag_rates.csv").is_file():
        rates = pl.read_csv(report_dir / "patient_flag_rates.csv")
        for mode in sorted(rates["mode"].unique()):
            part = rates.filter(pl.col("mode") == mode)
            contexts = sorted(part["X"].unique())
            if not contexts:
                continue
            assessed = part.filter(pl.col("inspection_applicable"))
            maximum = (
                (100 * assessed["patients_exact_flagged"] / assessed["patients"]).max()
                if not assessed.is_empty()
                else 0
            )
            upper = min(125, max(5, float(maximum or 0) * 1.25))
            fig, axes = plt.subplots(
                len(splits),
                len(contexts),
                figsize=(5 * len(contexts), 6),
                squeeze=False,
                constrained_layout=True,
            )
            for row, split in enumerate(splits):
                for column, context in enumerate(contexts):
                    cell = part.filter(
                        (pl.col("split") == split)
                        & (pl.col("X") == context)
                        & pl.col("inspection_applicable")
                    ).sort("Y")
                    axis = axes[row, column]
                    if cell.is_empty():
                        axis.text(
                            0.5,
                            0.5,
                            "No horizons with individual flags assessed",
                            ha="center",
                            va="center",
                            transform=axis.transAxes,
                            wrap=True,
                        )
                        axis.set_axis_off()
                        continue
                    count = cell["patients"].to_numpy()
                    exact = cell["patients_exact_flagged"].to_numpy()
                    low = cell["patients_flagged_low_diversity"].to_numpy()
                    known = cell["patients_diversity_unknown"].to_numpy() == 0
                    positions = np.arange(cell.height)
                    if known.all():
                        axis.bar(
                            positions,
                            100 * low / count,
                            label="Exact, low diversity",
                            color="#dd8452",
                        )
                        axis.bar(
                            positions,
                            100 * (exact - low) / count,
                            bottom=100 * low / count,
                            label="Exact, higher diversity",
                            color="#4c72b0",
                        )
                    else:
                        axis.bar(
                            positions,
                            100 * exact / count,
                            label="Exact; diversity unavailable",
                            color="grey",
                        )
                    for position, matches, low_count, total, available in zip(
                        positions, exact, low, count, known
                    ):
                        label = (
                            f"{matches}/{total}\nlow {low_count}/{matches}"
                            if available and matches
                            else f"{matches}/{total}\nlow n/a"
                            if available
                            else f"{matches}/{total}\nlow unknown"
                        )
                        axis.annotate(
                            label,
                            (position, 100 * matches / total),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7,
                        )
                    axis.set_xticks(positions, cell["Y"].to_list())
                    axis.set(
                        xlabel="Horizon (sequence positions)",
                        ylabel="Exact matches / evaluated patients (%)",
                        title=f"{split}, X={context}",
                        ylim=(0, upper),
                    )
                    axis.grid(axis="y", alpha=0.2)
                    axis.legend(fontsize=7)
            fig.suptitle(
                scope
                + f" / {mode} — exact-match counts and low diversity\nAnnotations: exact/evaluated; low-diversity exact/exact. No individual flags below the configured inspection horizon."
            )
            fig.savefig(output_dir / f"{mode}_exact_diversity.png", dpi=170)
            plt.close(fig)

            if {"percent_low_diversity", "percent_flagged"} <= set(part.columns):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                colors = dict(
                    zip(contexts, plt.cm.viridis(np.linspace(0.1, 0.9, len(contexts))))
                )
                for axis, metric, label in zip(
                    axes,
                    ("percent_low_diversity", "percent_flagged"),
                    (
                        "Low-diversity continuations / evaluated (%)",
                        "Any individual flag / evaluated (%)",
                    ),
                ):
                    for context in contexts:
                        for split, style in zip(splits, ("-", "--")):
                            cell = part.filter(
                                (pl.col("X") == context) & (pl.col("split") == split)
                            ).sort("Y")
                            if metric == "percent_flagged":
                                cell = cell.filter(pl.col("inspection_applicable"))
                            else:
                                cell = cell.filter(
                                    pl.col("patients_diversity_unknown") == 0
                                )
                            if not cell.is_empty():
                                axis.plot(
                                    cell["Y"],
                                    cell[metric],
                                    style,
                                    marker="o",
                                    color=colors[context],
                                    label=f"X={context}, {split}",
                                )
                    axis.set(
                        xlabel="Horizon (sequence positions)",
                        ylabel=label,
                        ylim=(-2, 102),
                    )
                    axis.set_xscale("log", base=2)
                    axis.grid(alpha=0.2)
                if axes[0].get_legend_handles_labels()[0]:
                    axes[0].legend(fontsize=7)
                fig.suptitle(
                    scope
                    + f" / {mode} — diversity and individual review rates (not percentages memorized)"
                )
                fig.savefig(output_dir / f"{mode}_diversity_rates.png", dpi=170)
                plt.close(fig)

    if (report_dir / "diversity_exclusions.csv").is_file():
        excluded = pl.read_csv(report_dir / "diversity_exclusions.csv")
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for part in excluded.partition_by("mode", "split", "X", maintain_order=True):
            first = part.row(0, named=True)
            part = part.sort("Y")
            axis.plot(
                part["Y"],
                100 * part["patients_retained"] / part["patients_before"],
                marker="o",
                label=f"{first['mode']}/{first['split']}, X={first['X']}",
            )
        axis.set(
            xlabel="Horizon (sequence positions)",
            ylabel="Patients retained after diversity filter (%)",
            ylim=(-2, 102),
            title=scope,
        )
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.savefig(output_dir / "diversity_retention.png", dpi=170)
        plt.close(fig)

    if include_patient_distributions and (report_dir / "patient_losses.csv").is_file():
        # Read only split/loss columns; never IDs, codes or trajectories. No resampling.
        losses = pl.read_csv(
            report_dir / "patient_losses.csv", columns=["split", "mean_next_code_loss"]
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        values = {
            split: losses.filter(pl.col("split") == split)[
                "mean_next_code_loss"
            ].to_numpy()
            for split in splits
        }
        all_values = np.concatenate(list(values.values()))
        if len(all_values) and np.isfinite(all_values).all():
            bins = np.histogram_bin_edges(all_values, bins=50)
            for split, scores in values.items():
                if not len(scores):
                    continue
                axes[0].hist(
                    scores,
                    bins=bins,
                    histtype="step",
                    density=True,
                    label=f"{split}, n={len(scores)}",
                )
                quantiles = np.linspace(0, 1, max(2, min(501, len(scores))))
                axes[1].plot(np.quantile(scores, quantiles), quantiles, label=split)
            axes[0].set(xlabel="Mean next-code loss per patient", ylabel="Density")
            axes[1].set(
                xlabel="Mean next-code loss per patient",
                ylabel="Empirical cumulative fraction (quantile display)",
            )
            for axis in axes:
                axis.legend(fontsize=8)
                axis.grid(alpha=0.2)
            note = (
                " / unfiltered loss comparison"
                if manifest.get("diversity_filter")
                else ""
            )
            fig.suptitle(
                scope
                + note
                + " — teacher-forced loss distributions; no new attack or threshold"
            )
            fig.savefig(output_dir / "loss_distributions.png", dpi=170)
        plt.close(fig)
