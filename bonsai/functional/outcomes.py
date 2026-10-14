import polars as pl

from bonsai.functional.actions import action_router
from bonsai.functional.features import compute_abspos


def fill_nans_with_sampled(dates, seed=None):
    if dates.is_null().all():
        raise ValueError("No non-NaN indexing dates found")

    return dates.fill_null(
        dates.drop_nulls().sample(dates.len(), with_replacement=True, seed=seed)
    )


def get_index_dates(df: pl.DataFrame, cfg, seed: int) -> pl.DataFrame:
    """Index dates for one shard. sample: missing dates are drawn from this shard's observed ones."""
    index_dates = action_router(df, cfg.conditional_criteria)
    if cfg.missing == "drop":
        return index_dates.drop_nulls("index_date")

    index_dates = (
        df.select("subject_id")
        .unique()
        .join(index_dates, on="subject_id", how="left")
        .sort("subject_id")  # fixed order -> reproducible sampling
    )
    return index_dates.with_columns(
        index_date=fill_nans_with_sampled(index_dates["index_date"], seed=seed)
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
