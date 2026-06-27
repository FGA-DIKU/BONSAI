from typing import List, Literal, Optional, Set, Tuple
from datetime import datetime, timedelta
import warnings

import polars as pl

DuplicateSubjectPolicy = Literal["all", "first_lowest_censor_abspos"]


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: List, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Returns the first row (priority based on condition order) for each subject that matches the conditions"""
    df = df.with_columns(_prio=pl.lit(None).cast(pl.Int32))
    row_mask = pl.lit(False)
    subject_sets = []

    for i, cond in enumerate(conditions):
        cond_expr = pl.col(cond["col"]).is_in(cond["vals"])
        row_mask = row_mask | cond_expr
        df = df.with_columns(
            _prio=pl.when(cond_expr & pl.col("_prio").is_null())
            .then(pl.lit(i))
            .otherwise(pl.col("_prio"))
        )
        subject_sets.append(
            set(df.filter(cond_expr).get_column("subject_id").to_list())
        )

    if dependence == "independent":
        matched_subjects = set.union(*subject_sets)
    elif dependence == "dependent":
        matched_subjects = set.intersection(*subject_sets)
    else:
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    res = df.filter(pl.col("subject_id").is_in(list(matched_subjects)) & row_mask)
    res = (
        res.sort(["_prio", "time"]).group_by("subject_id", maintain_order=True).first()
    )
    return res.drop("_prio")


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


def resolve_duplicate_subject_outcomes(
    outcomes: pl.DataFrame,
    policy: DuplicateSubjectPolicy,
) -> pl.DataFrame:
    """Keep all rows, or one row per (split, subject_id) with lowest censor_abspos."""
    if policy == "all":
        return outcomes
    if policy != "first_lowest_censor_abspos":
        raise ValueError(
            "duplicate_subject_policy must be 'all' or 'first_lowest_censor_abspos', "
            f"not {policy!r}"
        )
    if outcomes.is_empty():
        return outcomes

    group_cols = (
        ["split", "subject_id"] if "split" in outcomes.columns else ["subject_id"]
    )
    has_dupes = (
        outcomes.group_by(group_cols)
        .len()
        .filter(pl.col("len") > 1)
        .height
        > 0
    )
    if not has_dupes:
        return outcomes

    n_before = outcomes.height
    deduped = (
        outcomes.sort([*group_cols, "censor_abspos"], nulls_last=True)
        .group_by(group_cols, maintain_order=True)
        .first()
    )
    print(
        f"duplicate_subject_policy={policy!r}: kept {deduped.height:_} of {n_before:_} "
        f"outcome rows ({n_before - deduped.height:_} dropped; lowest censor_abspos per "
        f"{'split and ' if 'split' in group_cols else ''}subject_id)",
        flush=True,
    )
    return deduped


def warn_duplicate_subject_outcomes(
    outcomes: pl.DataFrame,
    *,
    split_name: Optional[str] = None,
    max_examples: int = 5,
) -> None:
    dupes = outcomes.filter(pl.col("subject_id").is_duplicated())
    if dupes.is_empty():
        return

    n_rows = dupes.height
    n_subjects = dupes["subject_id"].n_unique()
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
    examples = dupes.select(example_cols).sort("subject_id").head(max_examples)
    example_lines = str(examples)
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
    split_name: Optional[str] = None,
) -> List[dict]:
    """One record per input row. Supports multiple outcomes per subject_id."""
    if outcomes.is_empty():
        return []

    time_delta_hours = (pl.col("outcome_date") - pl.col("index_date")).dt.total_hours()
    outcomes_in_prediction_window = pl.lit(n_hours_start_include) <= time_delta_hours
    if n_hours_end_include is not None:
        outcomes_in_prediction_window = outcomes_in_prediction_window & (
            time_delta_hours <= pl.lit(n_hours_end_include)
        )

    labeled = outcomes.with_columns(
        label=outcomes_in_prediction_window.fill_null(False).cast(pl.Int64)
    )
    warn_duplicate_subject_outcomes(labeled, split_name=split_name)
    return labeled.select("subject_id", "label", "censor_abspos").to_dicts()


def outcomes_to_frame(outcomes: List[dict]) -> pl.DataFrame:
    return pl.DataFrame(outcomes)


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


def split_and_binarize_outcomes(
    outcomes: pl.DataFrame,
    train_key: str,
    val_key: str,
    test_key: str,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_outcomes = outcomes.filter(pl.col("split") == train_key)
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


def print_outcome_split_summary(splits: dict[str, List[dict]]) -> None:
    for split_name, outcomes in splits.items():
        n_rows = len(outcomes)
        n_subjects = len({outcome["subject_id"] for outcome in outcomes})
        n_positives = sum(outcome["label"] for outcome in outcomes)
        positive_pct = 100.0 * n_positives / n_rows if n_rows else 0.0
        print(
            f"{split_name}: {n_rows:_} outcome rows, {n_subjects:_} unique subjects, "
            f"{n_positives:_} positives ({positive_pct:.2f}%)"
        )
