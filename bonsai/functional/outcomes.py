from typing import List, Literal, Optional, Dict, Tuple
from datetime import datetime, timedelta
import polars as pl


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: List, dependence: Literal["independent", "dependent"]
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


def warn_duplicate_subject_outcomes(
    outcomes: pd.DataFrame,
    *,
    split_name: Optional[str] = None,
    max_examples: int = 5,
) -> None:
    dup_mask = outcomes.duplicated(subset=["subject_id"], keep=False)
    if not dup_mask.any():
        return

    n_rows = int(dup_mask.sum())
    n_subjects = outcomes.loc[dup_mask, "subject_id"].nunique()
    split_part = f" in split {split_name!r}" if split_name is not None else ""

    example_cols = [
        c
        for c in [
            "subject_id",
            "index_date",
            "outcome_date",
            "censor_abspos",
            "censor_date",
            "label",
        ]
        if c in outcomes.columns
    ]
    examples = (
        outcomes.loc[dup_mask, example_cols].sort_values("subject_id").head(max_examples)
    )
    example_lines = examples.to_string(index=False)
    message = (
        f"WARNING: Found {n_rows} outcome rows for {n_subjects} subject_id(s) "
        f"with duplicates{split_part}. Each row will be used as a separate sample.\n"
        f"Example rows:\n{example_lines}"
    )
    print(message, flush=True)
    warnings.warn(
        message.removeprefix("WARNING: "),
        UserWarning,
        stacklevel=3,
    )


def binarize_outcomes(
    outcomes: pl.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
<<<<<<< HEAD
<<<<<<< HEAD
) -> Dict[int, dict]:
    time_delta_datetime = pl.col("outcome_date") - pl.col("index_date")
    time_delta_hours = time_delta_datetime.dt.total_hours()
=======
=======
    split_name: Optional[str] = None,
>>>>>>> 22a3328 (warning on multiple outcomes)
) -> List[dict]:
    """One record per input row. Supports multiple outcomes per subject_id."""
    if outcomes.empty:
        return []

    outcomes = outcomes.copy()
    time_delta_datetime = outcomes["outcome_date"] - outcomes["index_date"]
    time_delta_hours = time_delta_datetime.dt.days * 24
>>>>>>> c046c73 (try xgb and multiple outcomes)

    outcomes_in_prediction_window = pl.lit(n_hours_start_include) <= time_delta_hours
    if n_hours_end_include is not None:
        outcomes_in_prediction_window = outcomes_in_prediction_window & (
            time_delta_hours <= pl.lit(n_hours_end_include)
        )

<<<<<<< HEAD
    outcomes = outcomes.with_columns(
        label=outcomes_in_prediction_window.fill_null(False).cast(pl.Int64)
    )
=======
    outcomes["label"] = outcomes_in_prediction_window.astype(int)
    warn_duplicate_subject_outcomes(outcomes, split_name=split_name)
    return outcomes[["subject_id", "label", "censor_abspos"]].to_dict(orient="records")


def outcomes_to_frame(outcomes: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(outcomes)


def outcome_subject_ids(outcomes: List[dict]) -> Set[int]:
    return {outcome["subject_id"] for outcome in outcomes}


def expand_subjects_for_outcomes(
    subjects: List[dict], outcomes: List[dict]
) -> Tuple[List[dict], List[dict]]:
    """One dataset row per outcome record; reuses subject event data when ids repeat."""
    subjects_by_id = {subject["subject_id"]: subject for subject in subjects}
    expanded_subjects: List[dict] = []
    expanded_outcomes: List[dict] = []
    for outcome in outcomes:
        subject_id = outcome["subject_id"]
        if subject_id in subjects_by_id:
            expanded_subjects.append(subjects_by_id[subject_id])
            expanded_outcomes.append(outcome)
    return expanded_subjects, expanded_outcomes
>>>>>>> c046c73 (try xgb and multiple outcomes)

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
<<<<<<< HEAD
) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, dict]]:
    train_outcomes = outcomes.filter(pl.col("split") == train_key)
=======
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_outcomes = outcomes[outcomes["split"] == train_key]
>>>>>>> c046c73 (try xgb and multiple outcomes)
    train_outcomes = binarize_outcomes(
        train_outcomes,
        n_hours_start_include,
        n_hours_end_include,
        split_name=train_key,
    )
    val_outcomes = outcomes.filter(pl.col("split") == val_key)
    val_outcomes = binarize_outcomes(
        val_outcomes,
        n_hours_start_include,
        n_hours_end_include,
        split_name=val_key,
    )
    test_outcomes = outcomes.filter(pl.col("split") == test_key)
    test_outcomes = binarize_outcomes(
        test_outcomes,
        n_hours_start_include,
        n_hours_end_include,
        split_name=test_key,
    )

    return train_outcomes, val_outcomes, test_outcomes
