from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from bonsai.functional.features import compute_abspos


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: list, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Returns the first row (priority based on condition order) for each subject that matches the conditions"""
    # Initialization
    df = df.with_columns(_prio=pl.lit(None).cast(pl.Int32))
    row_mask = pl.lit(False)
    subject_sets = []

    # Find matches (dataframe rows AND subject_ids) of conditions
    for i, cond in enumerate(conditions):
        cond_expr = pl.col(cond["col"]).is_in(cond["vals"])  # Rows that meet condition
        row_mask = row_mask | cond_expr  # OR operation
        df = df.with_columns(
            _prio=pl.when(cond_expr & pl.col("_prio").is_null())
            .then(pl.lit(i))
            .otherwise(pl.col("_prio"))
        )  # Set priority (to take first row later)
        subject_sets.append(
            set(df.filter(cond_expr).get_column("subject_id").to_list())
        )  # Get subjects that match condition

    # Toggle between any or all conditions met
    if dependence == "independent":
        matched_subjects = set.union(*subject_sets)  # Any condition met
    elif dependence == "dependent":  # TODO: Implement time_window
        matched_subjects = set.intersection(*subject_sets)  # All conditions met
    else:
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    # Get matched subjects AND rows
    res = df.filter(pl.col("subject_id").is_in(list(matched_subjects)) & row_mask)

    # Take first row based on `conditions` ordering
    res = (
        res.sort(["_prio", "time"]).group_by("subject_id", maintain_order=True).first()
    )
    res = res.drop("_prio")
    return res


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
    )["time"]


def fill_nans_with_sampled(dates):
    if dates.is_null().all():
        raise ValueError("No non-NaN indexing dates found")

    return dates.fill_null(
        dates.drop_nulls().sample(dates.len(), with_replacement=True)
    )


def binarize_outcomes(
    outcomes: pl.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: int | None = None,
) -> pl.DataFrame:
    time_delta_hours = (pl.col("outcome_date") - pl.col("index_date")).dt.total_hours()

    in_prediction_window = time_delta_hours >= n_hours_start_include

    if n_hours_end_include is not None:
        in_prediction_window &= time_delta_hours <= n_hours_end_include

    return outcomes.with_columns(
        label=in_prediction_window.fill_null(False).cast(pl.Int64)
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


def split_and_tte_outcomes(
    outcomes: pl.DataFrame,
    train_key: str,
    val_key: str,
    test_key: str,
    end_of_followup: dict | None,  # SHOULD BE A COLUMN
    n_hours_end_include: int | None = None,
):
    splits = split_outcomes(outcomes, train_key, val_key, test_key)

    return (
        finalize_outcomes(
            tte_outcomes(
                split,
                end_of_followup,
                n_hours_end_include,
            )
        )
        for split in splits
    )


def tte_outcomes(
    outcomes: pl.DataFrame,
    end_of_followup: dict | None,  # SHOULD BE A COLUMN
    n_hours_end_include: int | None = None,
) -> pl.DataFrame:
    """Adds time-to-event and event columns to outcomes dataframe"""
    if (
        "end_of_followup_date" not in outcomes.columns and end_of_followup is not None
    ):  # TODO: TEMPORARY
        outcomes = outcomes.with_columns(
            pl.datetime(**end_of_followup).alias("end_of_followup_date")
        )

    origin = pl.col("index_date")
    t_event = (pl.col("outcome_date") - origin).dt.total_hours()
    t_end = (pl.col("end_of_followup_date") - origin).dt.total_hours()

    has_event = t_event.is_not_null()
    t_cens = pl.min_horizontal(t_end, pl.lit(n_hours_end_include))

    return outcomes.with_columns(
        duration=pl.when(has_event & (t_event <= t_cens))
        .then(t_event)
        .otherwise(t_cens),
        label=(has_event & (t_event <= t_cens)).cast(pl.Int64),
    )
