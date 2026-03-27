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
    dob_rows = df.filter(
        (pl.col("code") == "DOB") & pl.col("time").is_not_null()
    ).select(["subject_id", "time"])

    if (dob_rows.height != dob_rows["subject_id"].n_unique()) or (
        dob_rows.height != df["subject_id"].n_unique()
    ):
        raise ValueError(
            f"Expected exactly one non-null DOB entry per subject_id. "
            f"Found {dob_rows.height} non-null DOB rows for "
            f"{dob_rows['subject_id'].n_unique()} unique subjects in DOB rows "
            f"and {df['subject_id'].n_unique()} unique subjects in the full DataFrame."
        )

    df = df.join(
        dob_rows.rename({"time": "dob_time"}),
        on="subject_id",
        how="left",
    )

    background_code = (
        pl.when(pl.col("time").is_null())
        .then(pl.lit("BACKGROUND//") + pl.col("code"))
        .otherwise(pl.col("code"))
        .alias("code")
    )

    background_time = (
        pl.when(pl.col("time").is_null())
        .then(pl.col("dob_time"))
        .otherwise(pl.col("time"))
        .alias("time")
    )

    df = df.with_columns(
        background_code,
        background_time,
    )

    df = df.drop("dob_time")

    return df, dob_rows


def compute_age() -> pl.Expr:
    """
    Compute age in years from columns:
      - time
      - dob_time
    """
    return (
        (
            pl.col("time").cast(pl.Datetime("ms"))
            - pl.col("dob_time").cast(pl.Datetime("ms"))
        ).dt.total_milliseconds()
        / (365.25 * 24 * 3600 * 1_000)
    ).cast(pl.Float64)


def compute_abspos(
    timestamps: Union[pl.Expr, pl.Series, datetime],
) -> Union[pl.Expr, pl.Series, float]:
    if isinstance(timestamps, datetime):
        return (
            pl.Series([timestamps])
            .cast(pl.Datetime("ms"))
            .dt.timestamp("ms")
            .cast(pl.Float64)[0]
            / (3600 * 1_000)
        )

    if isinstance(timestamps, (pl.Expr, pl.Series)):
        return (
            timestamps
            .cast(pl.Datetime("ms"))
            .dt.timestamp("ms")
            .cast(pl.Float64)
            / (3600 * 1_000)
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