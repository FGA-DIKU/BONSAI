import pandas as pd
from typing import Tuple, Union
from datetime import datetime


def create_features(concepts: pd.DataFrame, logger) -> pd.DataFrame:
    """
    A function to create features from patient information and concepts DataFrames.
    We create background, death, age, absolute position, and segments features.
    """
    concepts, dob_info = create_background(concepts)
    # patient_info = create_patient_info(concepts)

    concepts = drop_invalids(concepts, logger) # Must be done post create_background

    features = concepts
    features["age"] = compute_age(features, dob_info)
    features = exclude_incorrect_event_ages(features, logger)
    features["abspos"] = compute_abspos(features["time"])

    features = features.sort_values(["subject_id", "time"]).reset_index(drop=True) # TODO: Needed in MEDS?
    features["segment"] = compute_segments(features)

    features = features[["subject_id", "code", "age", "abspos", "segment"]]

    return features

def create_background(concepts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """ Requires DOB (date of birth) token per person. Creates BACKGROUND//{var} tokens with time set to DOB time. """
    dob_rows = concepts[concepts["code"] == "DOB"]
    dob_info = dob_rows.set_index("subject_id")["time"]
    if len(dob_rows) != concepts["subject_id"].nunique():
        raise ValueError(f"Expected one DOB entry per subject_id, but found {len(dob_rows)} DOB entries for {concepts['subject_id'].nunique()} unique subject_ids.")

    bg_mask = concepts["time"].isna()

    concepts.loc[bg_mask, "code"] = "BACKGROUND//" + concepts.loc[bg_mask, "code"]
    concepts.loc[bg_mask, "time"] = concepts.loc[bg_mask, "subject_id"].map(dob_info)

    return concepts, dob_info

# def create_patient_info(concepts: pd.DataFrame, background: pd.DataFrame, logger) -> pd.DataFrame:
#     patient_info = pd.DataFrame({"subject_id": concepts["subject_id"].unique()})

#     # Extract birthdate
#     dob_rows = concepts[concepts["code"] == "DOB"]
#     patient_info["birthdate"] = patient_info["subject_id"].map(dob_rows.set_index("subject_id")["time"])

#     # Extract deathdate
#     dod_rows = concepts[concepts["code"] == "DOD"]
#     patient_info["deathdate"] = patient_info["subject_id"].map(dod_rows.set_index("subject_id")["time"])

#     # Extract background variables
#     if len(background) > 0:
#         bg_vars = background["code"].str.split("//").str[0].str.split("_").str[1].unique()
#         logger.info(f"Extracted background variables for patient info: {bg_vars}")
#         for var in bg_vars:
#             var_rows = background[background["code"].str.startswith(f"BACKGROUND_{var}")]
#             var_rows["value"] = var_rows["code"].str.split("//").str[1]
#             patient_info[var] = patient_info["subject_id"].map(var_rows.set_index("subject_id")["value"])

#     return patient_info

def compute_age(features: pd.DataFrame, dob_info: pd.Series) -> pd.Series:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'time' column.
        dob_info: Series with subject_id index and date of birth ´time´ values.
    Returns:
        pd.Series: age in years for each row in concepts
    """
    # Try to convert columns to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(features["time"]):
        features["time"] = pd.to_datetime(features["time"], errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(dob_info):
        dob_info = pd.to_datetime(dob_info, errors="coerce")

    return (features["time"] - features["subject_id"].map(dob_info)).dt.total_seconds() / (365.25 * 24 * 3600)


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

def drop_invalids(concepts: pd.DataFrame, logger) -> pd.DataFrame:   
    pre = len(concepts)
    concepts = concepts.dropna(subset=["subject_id", "code", "time"])
    if pre != len(concepts):
        logger.info(f"Dropped {pre - len(concepts)} rows with missing subject_id, code, or time")
    
    return concepts

def exclude_incorrect_event_ages(features: pd.DataFrame, logger) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    pre = len(features)
    features = features[(features["age"] >= -1) & (features["age"] <= 120)]
    if pre != len(features):
        logger.info(f"Dropped {pre - len(features)} rows with incorrect ages outside (-1,120) range")
    return features

