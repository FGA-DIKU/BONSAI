import polars as pl
import logging
from datetime import datetime
from typing import Tuple, Union

def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create background, age, absolute position, and segment features.
    TODO: Death?
    """
    df, dob_info = create_background(df)

    df = drop_invalids(df)  # Must be done post create_background

    features = df.join(
        dob_info.rename({"time": "dob_time"}),
        on="subject_id",
        how="left",
    ).with_columns(
        compute_age().alias("age")
    )

    features = exclude_incorrect_event_ages(features)

    features = features.with_columns(
        compute_abspos(pl.col("time")).alias("abspos")
    )

    features = features.sort(["subject_id", "time"]).with_columns(
        compute_segments().alias("segment")
    )

    features = features.select(["subject_id", "code", "age", "abspos", "segment"])

    return features


def create_background(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Requires DOB token per person. Creates BACKGROUND//{var} tokens with time set to DOB time."""
    dob_rows = df.filter(pl.col("code") == "DOB").select(["subject_id", "time"])

    if dob_rows.height != df["subject_id"].n_unique():
        raise ValueError(
            f"Expected one DOB entry per subject_id, but found {dob_rows.height} DOB entries "
            f"for {df['subject_id'].n_unique()} unique subject_ids."
        )

    if dob_rows["time"].is_null().any():
        null_subjects = dob_rows.filter(pl.col("time").is_null())["subject_id"].to_list()
        raise ValueError(
            f"Found DOB entries with null time for subject_ids: {null_subjects}. "
            "A valid DOB time is required to assign background event timestamps."
        )

    df = (
        df.join(
            dob_rows.rename({"time": "dob_time"}),
            on="subject_id",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("time").is_null())
            .then(pl.lit("BACKGROUND//") + pl.col("code"))
            .otherwise(pl.col("code"))
            .alias("code"),
            pl.when(pl.col("time").is_null())
            .then(pl.col("dob_time"))
            .otherwise(pl.col("time"))
            .alias("time"),
        )
        .drop("dob_time")
    )

    return df, dob_rows


def compute_age() -> pl.Expr:
    """
    Compute age in years from columns:
      - time
      - dob_time
    """
    return (
        (
            pl.col("time").cast(pl.Datetime("us"))
            - pl.col("dob_time").cast(pl.Datetime("us"))
        ).dt.total_microseconds()
        / (365.25 * 24 * 3600 * 1_000_000)
    ).cast(pl.Float64)


def compute_abspos(
    timestamps: Union[pl.Expr, pl.Series, datetime],
) -> Union[pl.Expr, float, pl.Series]:
    if isinstance(timestamps, datetime):
        ts = pl.Series([timestamps]).cast(pl.Datetime("us"))
        return ts.dt.timestamp("us")[0] / (3600 * 1_000_000)

    if isinstance(timestamps, pl.Series):
        ts = timestamps.cast(pl.Datetime("us"))
        return ts.dt.timestamp("us").cast(pl.Float64) / (3600 * 1_000_000)

    if isinstance(timestamps, pl.Expr):
        return (
            timestamps.cast(pl.Datetime("us")).dt.timestamp("us").cast(pl.Float64)
            / (3600 * 1_000_000)
        )

    raise TypeError(
        "Invalid type for timestamps, only pl.Expr, pl.Series, and datetime are supported."
    )


def compute_segments() -> pl.Expr:
    return (
        (pl.col("time") != pl.col("time").shift(1))
        .fill_null(True)
        .cast(pl.Int64)
        .cum_sum()
        .over("subject_id")
    )


def drop_invalids(df: pl.DataFrame) -> pl.DataFrame:
    pre = len(df)
    df = df.drop_nulls(["subject_id", "code", "time"])
    if pre != len(df):
        logging.info(
            f"drop_invalids: Dropped {pre - len(df)} rows with missing subject_id, code, or time"
        )
    return df


def exclude_incorrect_event_ages(features: pl.DataFrame) -> pl.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)."""
    pre = len(features)
    features = features.filter((pl.col("age") >= -1) & (pl.col("age") <= 120))
    if pre != len(features):
        logging.info(
            f"exclude_incorrect_event_ages: Dropped {pre - len(features)} rows with incorrect ages outside (-1,120) range"
        )
    return features