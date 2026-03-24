from typing import List, Literal, Optional, Dict, Tuple
from datetime import datetime
import polars as pl


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: List, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Returns the first row (priority based on condition order) for each subject that matches the conditions"""
    # Initialization
    df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("_prio"))
    row_mask = pl.lit(False)
    subject_sets = []

    # Find matches (dataframe rows AND subject_ids) of conditions
    for i, cond in enumerate(conditions):
        cond_expr = pl.col(cond["col"]).is_in(cond["vals"])  # Rows that meet condition
        row_mask = row_mask | cond_expr  # OR operation
        df = df.with_columns(
            pl.when(cond_expr).then(pl.lit(i)).otherwise(pl.col("_prio")).alias("_prio")
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
        res.sort("_prio")
        .group_by("subject_id", maintain_order=True)
        .first()
    )
    res = res.drop("_prio")
    return res


def get_date_from_absolute_date(absolute_date):
    assert absolute_date is not None
    return datetime(**absolute_date)


def get_date_from_relative_date(relative_dates, relative_hour_shift):
    assert relative_dates is not None
    assert relative_hour_shift is not None
    return relative_dates + pl.duration(hours=relative_hour_shift)


def get_date_from_exposure_date(subjects, df, dependence, conditions):
    assert subjects is not None
    assert df is not None
    assert dependence is not None
    assert conditions is not None
    result = get_subject_first_row_for_conditions(
        df, conditions=conditions, dependence=dependence
    )
    return subjects.join(
        result.select(["subject_id", "time"]), on="subject_id", how="left"
    )["time"]


def fill_nans_with_sampled(dates):
    if dates.drop_nulls().len() == 0:
        raise ValueError("No non-NaN indexing dates found")
    # Randomly sample from non-null
    samples = dates.drop_nulls().sample(
        n=dates.is_null().sum(), with_replacement=True
    )  # TODO: Add seed?
    sample_iter = iter(samples.to_list())
    filled = [next(sample_iter) if x is None else x for x in dates.to_list()]
    return pl.Series(dates.name, filled)


def binarize_outcomes(
    outcomes: pl.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
) -> Dict[int, dict]:
    time_delta_datetime = pl.col("outcome_date") - pl.col("index_date")
    time_delta_hours = time_delta_datetime.dt.total_hours()

    outcomes_in_prediction_window = pl.lit(n_hours_start_include) <= time_delta_hours
    if n_hours_end_include is not None:
        outcomes_in_prediction_window = outcomes_in_prediction_window & (
            time_delta_hours <= pl.lit(n_hours_end_include)
        )

    outcomes = outcomes.with_columns(
        outcomes_in_prediction_window
        .fill_null(False)
        .cast(pl.Int64)
        .alias("label")
    )

    rows = outcomes.select(["subject_id", "label", "censor_abspos"]).to_dicts()
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
) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, dict]]:
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