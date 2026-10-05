import polars as pl

from bonsai.functional.actions import absolute_date, first_match, relative_date
from bonsai.functional.conditions import get_subject_first_row_for_conditions
from bonsai.functional.features import compute_abspos


def fill_nans_with_sampled(dates, seed=None):
    if dates.is_null().all():
        raise ValueError("No non-NaN indexing dates found")

    return dates.fill_null(
        dates.drop_nulls().sample(dates.len(), with_replacement=True, seed=seed)
    )


def binarize_outcomes(
    outcomes: pl.DataFrame,
    start_include: dict,
    end_include: dict | None = None,
) -> pl.DataFrame:
    window_start = pl.col("index_date") + pl.duration(**start_include)
    has_outcome = pl.col("outcome_date").is_not_null()
    outcomes = outcomes.filter(~(has_outcome & (pl.col("outcome_date") < window_start)))

    if outcomes.select((pl.col("censor_date") > window_start).any()).item():
        raise ValueError(
            "censor_date is after the prediction window start; outcomes would leak into the input"
        )

    in_window = has_outcome
    if end_include is not None:
        in_window &= pl.col("outcome_date") <= pl.col("index_date") + pl.duration(
            **end_include
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
    start_include: dict,
    end_include: dict | None = None,
):
    splits = split_outcomes(outcomes, train_key, val_key, test_key)

    return (
        finalize_outcomes(
            binarize_outcomes(
                split,
                start_include,
                end_include,
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
            "label": row["label"],
            "censor_abspos": row["censor_abspos"],
        }
        for row in outcomes.to_dicts()
    }


def get_outcome_dates(df, action, conditional_criteria):
    if action == "first_match":
        outcome_dates = first_match(df, conditional_criteria).rename(
            {"time": "outcome_date"}
        )
    else:
        raise ValueError(f"get_outcome_dates action={action} is not yet supported")
    return outcome_dates


def get_index_dates(df, action, conditional_criteria):
    cohort = df.select("subject_id").unique()
    # Assign index dates
    if action == "absolute_date":
        index_dates = cohort.with_columns(
            index_date=absolute_date(conditional_criteria["absolute_date"])
        )
    elif action == "relative":
        index_dates = cohort.with_columns(
            index_date=relative_date(
                dates=pl.col("outcome_date"),
                relative_shift=conditional_criteria["relative_shift"],
            )
        )
    elif action == "exposure":
        index_dates = first_match(df, conditional_criteria).rename(
            {"time": "index_date"}
        )
    else:
        raise ValueError(f"get_index_dates action={action} is not yet supported")

    index_dates = cohort.join(index_dates, on="subject_id", how="left")
    return index_dates


def get_cohort(df, conditional_criteria) -> set:
    # TODO: This doesn't match the `first_row` logic in actions
    # TODO: This needs to reference index_date!
    criterion_population = set(df["subject_id"].to_list())
    for criterion in conditional_criteria:
        if criterion.action == "exclude":
            exclude_df = get_subject_first_row_for_conditions(
                df, criterion.conditions, criterion.dependence
            )
            criterion_population -= set(exclude_df["subject_id"].to_list())
        elif criterion.action == "include":
            include_df = get_subject_first_row_for_conditions(
                df, criterion.conditions, criterion.dependence
            )
            criterion_population &= set(include_df["subject_id"].to_list())
        else:
            raise ValueError(
                f"Criterion action can only be [include, exclude], not {criterion.action}"
            )
    return criterion_population


def get_censor_dates(df, action, conditional_criteria):
    assert len(df) == len(df["subject_id"].unique()), (
        "Duplicate subject_ids found in df"
    )
    if action == "relative":
        censor_dates = df.with_columns(
            censor_date=relative_date(
                dates=pl.col("index_date"),
                relative_shift=conditional_criteria["relative_shift"],
            )
        )
    else:
        raise ValueError(f"get_censor_dates action={action} is not yet supported")
    return censor_dates
