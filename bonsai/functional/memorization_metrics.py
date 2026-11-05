"""Patient scores, uncertainty, coverage and inspection triggers shared by both runs.

Generation and cohort selection live elsewhere; these functions only consume
predictions or score tables. Every required mean-excess interval must include
zero. Missing support is inconclusive; patient audit flags are separate evidence.
"""

import math

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

MODES = ("code_only", "true_features", "forward_fill")

SCORES = ("exact_match", "overlap", "rare_codes")

INSPECTION_HORIZON = 32

PARTIAL_REPRODUCTION = 0.9

MIN_UNIQUE_OVERLAP_TOKENS = (
    3  # Distinct predicted non-special codes; SEP does not count.
)

LOSS_COLUMNS = [
    "patients_train",
    "patients_unseen",
    "loss_train",
    "loss_unseen",
    "auc",
    "auc_lower",
    "auc_upper",
]

EXCESS_COLUMNS = [
    "mode",
    "score",
    "X",
    "Y",
    "patients_train",
    "patients_unseen",
    "train",
    "unseen",
    "excess",
    "lower",
    "upper",
]


def count_reproduced(generated, truth, codes: set):
    """(real events reproduced, real events counted), over real codes in ``codes``.

    Order does not matter, but each written code stands in for one real event
    only.
    """
    used = np.zeros(len(generated), dtype=bool)
    reproduced = counted = 0
    for code in truth:
        if code not in codes:
            continue
        counted += 1
        matches = (generated == code) & ~used
        if matches.any():
            used[matches.argmax()] = True
            reproduced += 1
    return reproduced, counted


def score_patient(generated, truth, countable: set, rare: set) -> dict:
    """One patient's three scores (NaN when there is nothing to count)."""
    hits, n = count_reproduced(generated, truth, countable)
    rare_hits, rare_n = count_reproduced(generated, truth, rare)
    return {
        "exact_match": float((generated == truth).all()) if len(truth) else math.nan,
        "overlap": hits / n if n else math.nan,
        "rare_codes": rare_hits / rare_n if rare_n else math.nan,
        "unique_predicted_tokens": len(set(generated.tolist()) & countable),
        "unique_reference_tokens": len(set(truth.tolist()) & countable),
    }


def grid_coverage(
    scores,
    prompts,
    horizons,
    splits,
    context,
    min_patients,
    minimum_diversity=None,
    modes=MODES,
):
    """Every requested cell, including unsupported/undersampled cells; no silent skips."""
    counts = {
        (row["mode"], row["split"], row["X"], row["Y"]): row["len"]
        for row in scores.group_by("mode", "split", "X", "Y")
        .len()
        .iter_rows(named=True)
    }
    rows = []
    for mode in modes:
        for split in splits:
            for x in sorted(set(prompts)):
                for y in sorted(set(horizons)):
                    n = counts.get((mode, split, x, y), 0)
                    status = (
                        "not_applicable_diversity"
                        if minimum_diversity is not None and y < minimum_diversity
                        else "unsupported_context"
                        if x + y > context + 1
                        else "too_few_patients"
                        if n < min_patients
                        else "complete"
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "split": split,
                            "X": x,
                            "Y": y,
                            "patients": n,
                            "status": status,
                        }
                    )
    return pl.DataFrame(rows)


def patient_tails(
    scores: pl.DataFrame, minimum_unique=MIN_UNIQUE_OVERLAP_TOKENS
) -> pl.DataFrame:
    """Describe per-patient tails without exporting individual scores or identifiers.

    p95/max and counts scoring 1 expose concentrated reproduction hidden by means.
    These are descriptive summaries, not a calibrated membership attack or gate.
    """
    # Older score tables lack diversity counts; retain their original flag rule.
    if "unique_predicted_tokens" not in scores.columns:
        scores = scores.with_columns(
            pl.lit(minimum_unique).alias("unique_predicted_tokens")
        )
    values = scores.unpivot(
        on=SCORES,
        index=["mode", "split", "X", "Y", "unique_predicted_tokens"],
        variable_name="score",
        value_name="value",
    ).filter(pl.col("value").is_finite())
    return (
        values.group_by("mode", "split", "X", "Y", "score")
        .agg(
            pl.len().alias("patients"),
            pl.col("value").median().alias("median"),
            pl.col("value").quantile(0.95, interpolation="linear").alias("p95"),
            pl.col("value").max().alias("max"),
            (pl.col("value") == 1).sum().alias("patients_scoring_1"),
            (pl.col("value") >= PARTIAL_REPRODUCTION)
            .sum()
            .alias("patients_scoring_ge_90pct"),
            (
                (pl.col("score") == "overlap")
                & (pl.col("value") >= PARTIAL_REPRODUCTION)
                & (pl.col("unique_predicted_tokens") >= minimum_unique)
            )
            .sum()
            .alias("patients_high_overlap_diverse"),
        )
        .with_columns(
            (100 * pl.col("patients_scoring_1") / pl.col("patients")).alias(
                "percent_scoring_1"
            ),
            (100 * pl.col("patients_scoring_ge_90pct") / pl.col("patients")).alias(
                "percent_scoring_ge_90pct"
            ),
        )
        .sort("mode", "split", "X", "Y", "score")
    )


def patient_flag_rates(scores, minimum_unique=MIN_UNIQUE_OVERLAP_TOKENS):
    """Per-cell union of the patient-level triggers: never count one patient twice.

    Denominator is evaluated patients, not the full cohort. Below the inspection
    horizon flags are not assessed; percentages are null, rather than reassuring 0.
    """
    applicable = pl.col("Y") >= INSPECTION_HORIZON
    exact = (pl.col("exact_match") == 1).fill_null(False)
    overlap = (
        pl.col("overlap").is_finite() & (pl.col("overlap") >= PARTIAL_REPRODUCTION)
    ).fill_null(False)
    if "unique_predicted_tokens" in scores.columns:
        overlap &= pl.col("unique_predicted_tokens") >= minimum_unique
    diversity = (
        pl.col("unique_predicted_tokens")
        if "unique_predicted_tokens" in scores.columns
        else pl.lit(None, dtype=pl.Int64)
    )
    return (
        scores.group_by("mode", "split", "X", "Y")
        .agg(
            pl.len().alias("patients"),
            (diversity < minimum_unique)
            .fill_null(False)
            .sum()
            .alias("patients_low_diversity"),
            (
                diversity.is_null().sum()
                if "unique_predicted_tokens" in scores.columns
                else pl.len()
            ).alias("patients_diversity_unknown"),
            ((exact | overlap) & applicable & (diversity < minimum_unique))
            .fill_null(False)
            .sum()
            .alias("patients_flagged_low_diversity"),
            ((exact | overlap) & applicable).sum().alias("patients_flagged"),
            (exact & applicable).sum().alias("patients_exact_flagged"),
            (overlap & applicable).sum().alias("patients_overlap_flagged"),
        )
        .with_columns(
            applicable.alias("inspection_applicable"),
            pl.lit(minimum_unique).alias("minimum_unique_overlap_tokens"),
            (100 * pl.col("patients_low_diversity") / pl.col("patients")).alias(
                "percent_low_diversity"
            ),
            pl.when(applicable)
            .then(100 * pl.col("patients_flagged") / pl.col("patients"))
            .otherwise(None)
            .alias("percent_flagged"),
            pl.when(applicable)
            .then(100 * pl.col("patients_exact_flagged") / pl.col("patients"))
            .otherwise(None)
            .alias("percent_exact_flagged"),
            pl.when(pl.col("patients_exact_flagged") > 0)
            .then(
                100
                * pl.col("patients_flagged_low_diversity")
                / pl.col("patients_exact_flagged")
            )
            .otherwise(None)
            .alias("percent_low_diversity_among_exact"),
        )
        .sort("mode", "split", "X", "Y")
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion; not empty when no patient matches."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def difference_in_means(train: np.ndarray, unseen: np.ndarray, is_proportion: bool):
    """Mean of ``train`` minus mean of ``unseen`` with a 95% interval: Newcombe
    (1998, method 10) for exact match (yes/no), otherwise normal (Welch)."""
    p1, p2 = train.mean(), unseen.mean()
    if is_proportion:
        l1, u1 = wilson_interval(int(train.sum()), len(train))
        l2, u2 = wilson_interval(int(unseen.sum()), len(unseen))
        below, above = math.hypot(p1 - l1, u2 - p2), math.hypot(u1 - p1, p2 - l2)
        return p1 - p2, p1 - p2 - below, p1 - p2 + above
    variance = train.var(ddof=1) / len(train) + unseen.var(ddof=1) / len(unseen)
    if variance == 0:
        # All-zero overlap must not imply zero uncertainty about rare successes.
        # At boundaries, bound the probability of any unobserved nonzero/error
        # event; its mean contribution cannot exceed that probability. Allocate
        # 2.5% per group. Interior constants use a conservative bounded-score bound.
        def interval(mean, n):
            missing = -math.expm1(math.log(0.025) / n)
            if mean == 0:
                return 0.0, missing
            if mean == 1:
                return 1 - missing, 1.0
            radius = math.sqrt(math.log(80) / (2 * n))
            return max(0.0, mean - radius), min(1.0, mean + radius)

        lo1, hi1 = interval(p1, len(train))
        lo2, hi2 = interval(p2, len(unseen))
        return p1 - p2, lo1 - hi2, hi1 - lo2
    half = 1.96 * math.sqrt(variance)
    return p1 - p2, p1 - p2 - half, p1 - p2 + half


def excess_over_unseen(
    scores: pl.DataFrame, train_split: str, unseen_split: str
) -> pl.DataFrame:
    """Per score, X and Y: both groups' means and the excess, counting only the
    patients who have the score (e.g. rare_codes needs a rare code in the next Y)."""
    rows = []
    for (mode, x, y), part in scores.group_by("mode", "X", "Y"):
        in_train = part.filter(pl.col("split") == train_split)
        in_unseen = part.filter(pl.col("split") == unseen_split)
        for name in SCORES:
            train = in_train[name].drop_nans().to_numpy()
            unseen = in_unseen[name].drop_nans().to_numpy()
            if len(train) < 2 or len(unseen) < 2:
                continue
            is_proportion = name == "exact_match"
            excess = difference_in_means(train, unseen, is_proportion)
            counts = len(train), len(unseen), train.mean(), unseen.mean()
            rows.append((mode, name, x, y, *counts, *excess))
    excess = pl.DataFrame(rows, schema=EXCESS_COLUMNS, orient="row")
    return excess.sort("mode", "score", "X", "Y")


def membership_from_loss(
    train: np.ndarray, unseen: np.ndarray, *, allow_insufficient=False
) -> pl.DataFrame:
    """AUC for telling training from unseen patients by their loss (0.5 = not at
    all), with the Hanley and McNeil (1982) 95% interval."""
    n1, n2 = len(train), len(unseen)
    if min(n1, n2) < 2 and allow_insufficient:
        return pl.DataFrame(
            [
                (
                    n1,
                    n2,
                    float(train.mean()) if n1 else math.nan,
                    float(unseen.mean()) if n2 else math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
            ],
            schema=LOSS_COLUMNS,
            orient="row",
        )
    if min(n1, n2) < 2 or not np.isfinite(train).all() or not np.isfinite(unseen).all():
        raise ValueError(
            "Loss comparison needs at least two finite patient losses per group."
        )
    auc = roc_auc_score([1] * n1 + [0] * n2, -np.concatenate([train, unseen]))
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)
    half = 1.96 * math.sqrt(max(0.0, var) / (n1 * n2))
    row = (
        n1,
        n2,
        train.mean(),
        unseen.mean(),
        auc,
        max(0.0, auc - half),
        min(1.0, auc + half),
    )
    return pl.DataFrame([row], schema=LOSS_COLUMNS, orient="row")


def inspection_flags(tails: pl.DataFrame) -> pl.DataFrame:
    """Flag exact or diverse >=90% overlap at horizons >=32, even at n=1."""
    overlap_count = (
        "patients_high_overlap_diverse"
        if "patients_high_overlap_diverse" in tails.columns
        else "patients_scoring_ge_90pct"
    )
    return tails.filter(
        (pl.col("Y") >= INSPECTION_HORIZON)
        & (
            ((pl.col("score") == "exact_match") & (pl.col("patients_scoring_1") > 0))
            | ((pl.col("score") == "overlap") & (pl.col(overlap_count) > 0))
        )
    )


def assess_comparisons(excess, coverage, min_patients, *, loss=None, complete=True):
    """Assess the requested grid; optionally include the supplemental loss check.

    Both directions count as differences. Every required metric needs enough
    patients and a finite interval, including the rare-code denominator. A known
    difference takes precedence over incomplete sampling or missing comparisons.
    """
    enough = pl.min_horizontal("patients_train", "patients_unseen") >= min_patients
    finite = pl.col("lower").is_finite() & pl.col("upper").is_finite()
    # Header-only/empty tables can have Null key types; keep joins well-defined.
    assessed = excess.filter(enough & finite).cast(
        {"mode": pl.String, "score": pl.String, "X": pl.Int64, "Y": pl.Int64}
    )
    findings = assessed.filter((pl.col("lower") > 0) | (pl.col("upper") < 0))
    applicable = coverage.filter(pl.col("status") != "not_applicable_diversity")
    keys = ["mode", "X", "Y", "score"]
    expected = (
        applicable.select("mode", "X", "Y")
        .unique()
        .join(pl.DataFrame({"score": list(SCORES)}), how="cross")
    )
    missing = expected.join(assessed.select(keys), on=keys, how="anti")
    incomplete = applicable.filter(pl.col("status") != "complete")
    lines = [
        "Pointwise diagnostic intervals; passing does not establish equivalence or absence of memorization.",
        f"Generation comparisons assessed: {expected.height - missing.height}/{expected.height}; incomplete grid cells: {incomplete.height}.",
        f"Cells structurally inapplicable under the diversity filter: {coverage.height - applicable.height}.",
    ]
    loss_ready, loss_flag = True, False
    if loss is not None:
        row = loss.row(0, named=True)
        loss_ready = min(
            row["patients_train"], row["patients_unseen"]
        ) >= min_patients and all(
            math.isfinite(row[k]) for k in ("auc", "auc_lower", "auc_upper")
        )
        loss_flag = loss_ready and not row["auc_lower"] <= 0.5 <= row["auc_upper"]
        lines.append(
            f"Loss AUC: {row['auc']:.3f} ({row['auc_lower']:.3f} to {row['auc_upper']:.3f}); patients={row['patients_train']}/{row['patients_unseen']}."
        )
    if findings.height or loss_flag:
        status = "CHECK"
    elif (
        not complete
        or expected.is_empty()
        or missing.height
        or incomplete.height
        or not loss_ready
    ):
        status = "INCONCLUSIVE"
    else:
        status = "CHECKS PASSED"
    if missing.height:
        lines.append(
            "Missing/underpowered metrics (including rare-code denominators) are not counted as passed; see excess.csv and coverage.csv."
        )
    return status, findings, lines


def comparison_summary(excess, loss, min_patients, tails, coverage, *, complete=True):
    """Format the shared comparison assessment for supplemental reports."""
    status, findings, details = assess_comparisons(
        excess, coverage, min_patients, loss=loss, complete=complete
    )
    reasons = {
        "CHECK": "a train–comparison interval excludes its null; inspect the listed differences.",
        "INCONCLUSIVE": "some required comparisons lack adequate coverage or estimable intervals.",
        "CHECKS PASSED": "all required excess intervals include zero and loss AUC includes 0.5.",
    }
    lines = [f"{status}: {reasons[status]}", *details]
    for row in findings.iter_rows(named=True):
        lines.append(
            f"  {row['mode']} X={row['X']} Y={row['Y']} {row['score']}: excess={row['excess']:+.4f} ({row['lower']:+.4f}, {row['upper']:+.4f}); n={row['patients_train']}/{row['patients_unseen']}"
        )
    if tails is not None:
        lines.append(
            f"Patient-level reproduction triggers: {inspection_flags(tails).height} cells; audit trajectories separately. They do not independently determine this comparison gate."
        )
    return "\n".join(lines)
