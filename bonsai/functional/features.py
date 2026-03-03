import pandas as pd
from typing import Tuple, Union, Dict
from datetime import datetime
import logging


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A function to create features from a pandas DataFrame.
    We create background, , age, absolute position, and segments features.
    TODO: Death?
    """
    df, dob_info = create_background(df)

    df = drop_invalids(df)  # Must be done post create_background

    features = df
    features["age"] = compute_age(features, dob_info)
    features = exclude_incorrect_event_ages(features)

    features["abspos"] = compute_abspos(features["time"])

    features = features.sort_values(["subject_id", "time"]).reset_index(
        drop=True
    )  # TODO: Needed in MEDS?
    features["segment"] = compute_segments(features)

    features = features[["subject_id", "code", "age", "abspos", "segment"]]

    return features


def create_background(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Requires DOB (date of birth) token per person. Creates BACKGROUND//{var} tokens with time set to DOB time."""
    dob_rows = df[df["code"] == "DOB"]
    dob_info = dob_rows.set_index("subject_id")["time"]
    if len(dob_rows) != df["subject_id"].nunique():
        raise ValueError(
            f"Expected one DOB entry per subject_id, but found {len(dob_rows)} DOB entries for {df['subject_id'].nunique()} unique subject_ids."
        )

    bg_mask = df["time"].isna()

    df.loc[bg_mask, "code"] = "BACKGROUND//" + df.loc[bg_mask, "code"]
    df.loc[bg_mask, "time"] = df.loc[bg_mask, "subject_id"].map(dob_info)

    return df, dob_info


def compute_age(features: pd.DataFrame, dob_info: pd.Series) -> pd.Series:
    """
    Compute age in years for each row in features
    Parameters:
        features: Dataframe with 'time' column.
        dob_info: Series with subject_id index and date of birth ´time´ values.
    Returns:
        pd.Series: age in years for each row in features
    """
    # Try to convert columns to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(features["time"]):
        features["time"] = pd.to_datetime(features["time"], errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(dob_info):
        dob_info = pd.to_datetime(dob_info, errors="coerce")

    return (
        features["time"] - features["subject_id"].map(dob_info)
    ).dt.total_seconds() / (365.25 * 24 * 3600)


def compute_abspos(
    timestamps: Union[pd.Series, datetime],
) -> Union[pd.Series, float]:
    if isinstance(timestamps, datetime):
        return compute_abspos(pd.Series([timestamps])).iloc[0]

    if not isinstance(timestamps, pd.Series):
        raise TypeError(
            "Invalid type for timestamps, only pd.Series and datetime are supported."
        )

    # Convert timestamps to UTC (timezone-aware)
    timestamps = pd.to_datetime(
        timestamps, utc=True
    )  # ensure consistency across dataset
    # Remove the timezone information to get a timezone-naive series, necessary for the next step
    timestamps = timestamps.dt.tz_localize(None)
    # Cast to microsecond precision
    timestamps = timestamps.astype("datetime64[us]")
    # Convert microseconds to hours
    hours = (timestamps.astype("int64") // 10**6) / 3600
    return hours


def compute_segments(features: pd.DataFrame) -> pd.Series:
    return features.groupby("subject_id", sort=False)["time"].transform(
        lambda x: (x != x.shift()).cumsum()
    )


def drop_invalids(df: pd.DataFrame) -> pd.DataFrame:
    pre = len(df)
    df = df.dropna(subset=["subject_id", "code", "time"])
    if pre != len(df):
        logging.info(
            f"drop_invalids: Dropped {pre - len(df)} rows with missing subject_id, code, or time"
        )

    return df


def exclude_incorrect_event_ages(features: pd.DataFrame) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    pre = len(features)
    features = features[(features["age"] >= -1) & (features["age"] <= 120)]
    if pre != len(features):
        logging.info(
            f"exclude_incorrect_event_ages: Dropped {pre - len(features)} rows with incorrect ages outside (-1,120) range"
        )
    return features


def get_background_length(df: pd.DataFrame, vocabulary: Dict[str, int]):
    unique_background_tokens = set([i for i in vocabulary if i.startswith("BG_")])
    background_length = len(set(df) & unique_background_tokens)
    return background_length + 2  # +2 for [CLS] and [SEP] tokens
