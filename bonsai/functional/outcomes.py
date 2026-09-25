from datetime import datetime, timedelta
from typing import Literal, Optional

import polars as pl


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: list, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Earliest time each subject meets the definition: any condition (independent) or all (dependent)."""
    if dependence not in ("independent", "dependent"):
        raise ValueError(f"Dependence can only be [independent, dependent], not {dependence}")

    # Build conditions
    per_cond = [
        df.filter(pl.col(cond["col"]).is_in(cond["vals"]))
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
    n_hours_end_include: Optional[int] = None,
) -> dict[int, dict]:
    time_delta_datetime = pl.col("outcome_date") - pl.col("index_date")
    time_delta_hours = time_delta_datetime.dt.total_hours()

    outcomes_in_prediction_window = pl.lit(n_hours_start_include) <= time_delta_hours
    if n_hours_end_include is not None:
        outcomes_in_prediction_window = outcomes_in_prediction_window & (
            time_delta_hours <= pl.lit(n_hours_end_include)
        )

    outcomes = outcomes.with_columns(
        label=outcomes_in_prediction_window.fill_null(False).cast(pl.Int64)
    )

    rows = outcomes.select("subject_id", "label", "censor_abspos").to_dicts()
    return {
        row["subject_id"]: {
            "label": row["label"],
            "censor_abspos": row["censor_abspos"],
        }
        for row in rows
    }


def split_and_binarize_outcomes(
    outcomes,
    train_key: str,
    val_key: str,
    test_key: str,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    train_outcomes = outcomes.filter(pl.col("split") == train_key)
    train_outcomes = binarize_outcomes(
        train_outcomes, n_hours_start_include, n_hours_end_include
    )
    val_outcomes = outcomes.filter(pl.col("split") == val_key)
    val_outcomes = binarize_outcomes(
        val_outcomes, n_hours_start_include, n_hours_end_include
    )
    test_outcomes = outcomes.filter(pl.col("split") == test_key)
    test_outcomes = binarize_outcomes(
        test_outcomes, n_hours_start_include, n_hours_end_include
    )

    return train_outcomes, val_outcomes, test_outcomes
