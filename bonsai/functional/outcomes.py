from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from bonsai.functional.features import compute_abspos


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: list, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Earliest time each subject meets the definition: any condition (independent) or all (dependent)."""
    if dependence not in ("independent", "dependent"):
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    # Build conditions
    per_cond = [
        df.filter(
            pl.any_horizontal(
                pl.col(cond["col"]).str.starts_with(val) for val in cond["vals"]
            )
        )
        .group_by("subject_id")
        .agg(pl.col("time").min().alias(f"_time{i}"))
        for i, cond in enumerate(conditions)
    ]

    # Joins conditions
    how = "full" if dependence == "independent" else "inner"
    res = per_cond[0]
    for other in per_cond[1:]:
        res = res.join(other, on="subject_id", how=how, coalesce=True)

    # Find dependence time
    cols = [f"_time{i}" for i in range(len(conditions))]
    combine = pl.min_horizontal if dependence == "independent" else pl.max_horizontal

    return res.select("subject_id", combine(cols).alias("time"))


def get_date_from_absolute_date(absolute_date):
    assert absolute_date is not None
    return datetime(**absolute_date)


def get_date_from_relative_date(relative_dates, relative_hour_shift):
    assert relative_dates is not None
    assert relative_hour_shift is not None
    return relative_dates + timedelta(hours=relative_hour_shift)


def get_date_from_exposure_date(subjects, df, dependence, conditions):
    assert subjects is not None
    assert df is not None
    assert dependence is not None
    assert conditions is not None
    result = get_subject_first_row_for_conditions(
        df, conditions=conditions, dependence=dependence
    )
    return subjects.join(
        result.select("subject_id", "time"), on="subject_id", how="left"
    )


def fill_nans_with_sampled(dates, seed=None):
    if dates.is_null().all():
        raise ValueError("No non-NaN indexing dates found")

    return dates.fill_null(
        dates.drop_nulls().sample(dates.len(), with_replacement=True, seed=seed)
    )


def binarize_outcomes(
    outcomes: pl.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: int | None = None,
) -> pl.DataFrame:
    window_start = pl.col("index_date") + pl.duration(hours=n_hours_start_include)
    has_outcome = pl.col("outcome_date").is_not_null()
    outcomes = outcomes.filter(~(has_outcome & (pl.col("outcome_date") < window_start)))

    if outcomes.select((pl.col("censor_date") > window_start).any()).item():
        raise ValueError(
            "censor_date is after the prediction window start; outcomes would leak into the input"
        )

    in_window = has_outcome
    if n_hours_end_include is not None:
        in_window &= pl.col("outcome_date") <= pl.col("index_date") + pl.duration(
            hours=n_hours_end_include
        )
    return outcomes.with_columns(label=in_window.cast(pl.Int64))


def split_outcomes(
    outcomes: pl.DataFrame,
    train_key: str,
    val_key: str,
    test_key: str,
):
    return (
        outcomes.filter(pl.col("split") == split_key)
        for split_key in (train_key, val_key, test_key)
    )


def split_and_binarize_outcomes(
    outcomes: pl.DataFrame,
    train_key: str,
    val_key: str,
    test_key: str,
    n_hours_start_include: int,
    n_hours_end_include: int | None = None,
):
    splits = split_outcomes(outcomes, train_key, val_key, test_key)

    return (
        finalize_outcomes(
            binarize_outcomes(
                split,
                n_hours_start_include,
                n_hours_end_include,
            )
        )
        for split in splits
    )


def finalize_outcomes(outcomes: pl.DataFrame) -> dict[int, dict]:
    outcomes = outcomes.with_columns(
        censor_abspos=compute_abspos(pl.col("censor_date"))
    )
    return {
        row["subject_id"]: {
            key: value for key, value in row.items() if key != "subject_id"
        }
        for row in outcomes.to_dicts()
    }
