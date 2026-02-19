from datetime import datetime
from typing import Union
import uuid
import warnings
import numpy as np
import pandas as pd


def assign_index_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'index' and 'order' columns to ensure correct ordering.
    - The 'index' column represents the position of each row within its partition.
    - The 'order' column can be used for additional custom ordering if needed.
    - Both columns are initialized with 0 to ensure consistent behavior across partitions.
    Parameters:
        df: pd.DataFrame with 'PID' column.
    Returns:
        df with 'index' and 'order' columns.
    """
    df.loc[:, "index"] = df["index"].fillna(0)
    df.loc[:, "order"] = df["order"].fillna(0)
    return df


def create_abspos(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
    Returns:
        concepts with a new 'abspos' column
    """
    concepts["abspos"] = get_hours_since_epoch(concepts["time"])
    return concepts


def get_hours_since_epoch(
    timestamps: Union[pd.Series, datetime],
) -> Union[pd.Series, float]:
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([], dtype=float)
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

    elif isinstance(timestamps, datetime):
        return get_hours_since_epoch(pd.Series([timestamps])).iloc[0]
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series, list, and datetime are supported."
        )


def create_age_in_years(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'time' and 'birthdate' columns.
    Returns:
        pd.DataFrame: concepts with a new 'age' column
    """
    # Try to convert columns to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(concepts["time"]):
        concepts["time"] = pd.to_datetime(concepts["time"], errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(concepts["birthdate"]):
        concepts["birthdate"] = pd.to_datetime(concepts["birthdate"], errors="coerce")

    # Calculate age
    concepts["age"] = (
        concepts["time"] - concepts["birthdate"]
    ).dt.days // 365.25  # TODO: Not a good calculation

    return concepts


def create_background(concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create background concepts for each patient based on the static background variables in the dataframe.
    Sets the time of the background concepts to the birthdate of the patient.
    Expects 'DOB' concept to be present in the patients_info DataFrame.

    Args:
        concepts: DataFrame with columns 'subject_id', 'time', 'code'

    Returns:
        tuple: (updated_concepts_df, patient_info_df)
            - updated_concepts_df: concepts with background concepts updated and birthdate column added
            - patient_info_df: patient information with birthdate, deathdate, and background variables
    """
    # Create a copy to avoid modifying the original DataFrame
    concepts = concepts.copy()

    # Extract birthdates from DOB rows
    dob_rows = concepts[concepts["code"] == "DOB"]
    birthdates = dict(zip(dob_rows["subject_id"], dob_rows["time"]))
    concepts["birthdate"] = concepts["subject_id"].map(birthdates)
    if concepts["birthdate"].isna().any():
        raise ValueError("Some patients have no DOB")

    # Use boolean masking instead of index-based selection for background rows
    bg_mask = concepts["time"].isna()
    concepts.loc[bg_mask, "time"] = concepts.loc[bg_mask, "birthdate"]
    concepts.loc[bg_mask, "code"] = "BG_" + concepts.loc[bg_mask, "code"]

    # Use boolean masking for admission/discharge rows
    adm_mask = concepts["code"].str.contains("ADMISSION", na=False) | concepts[
        "code"
    ].str.contains("DISCHARGE", na=False)
    concepts.loc[adm_mask, "code"] = "ADM_" + concepts.loc[adm_mask, "code"]

    # Get the patient info
    patient_info = _create_patient_info(concepts)
    return concepts, patient_info


def _create_patient_info(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient information DataFrame from concepts.

    Args:
        concepts: DataFrame with patient concepts

    Returns:
        DataFrame with patient information including birthdate, deathdate, and background variables
    """
    # Get unique patients
    patients = concepts["subject_id"].unique()

    # Initialize patient info - handle empty case
    patient_info = pd.DataFrame({"subject_id": patients})

    # If no patients, return empty DataFrame with proper structure
    if len(patients) == 0:
        warnings.warn("No patients found in concepts")
        patient_info["birthdate"] = pd.Series([], dtype="datetime64[ns]")
        patient_info["deathdate"] = pd.Series([], dtype="datetime64[ns]")
        return patient_info

    # Fallback: extract from DOB codes
    dob_data = concepts[concepts["code"] == "DOB"]
    birthdate_map = dict(zip(dob_data["subject_id"], dob_data["time"]))
    patient_info["birthdate"] = patient_info["subject_id"].map(birthdate_map)

    # Extract death dates (DOD)
    dod_data = concepts[concepts["code"] == "DOD"]
    deathdate_map = dict(zip(dod_data["subject_id"], dod_data["time"]))
    patient_info["deathdate"] = patient_info["subject_id"].map(deathdate_map)

    # Extract background variables (those that start with BG_)
    bg_concepts = concepts[concepts["code"].str.startswith("BG_", na=False)]

    # Process background concepts if they exist
    if not bg_concepts.empty:
        bg_info = bg_concepts[["subject_id", "code"]].copy()

        # Split BG_ concepts into column_name and value, handling cases without "//"
        split_result = bg_info["code"].str.split("//", expand=True)

        # Ensure we always have at least 2 columns
        if split_result.shape[1] == 1:
            # No "//" separator found, add empty value column
            split_result[1] = None

        bg_info["column_name"] = split_result[0]
        bg_info["value"] = split_result[1]

        # Remove BG_ prefix from column names
        bg_info["column_name"] = bg_info["column_name"].str.replace("BG_", "")

        # Filter out rows without proper column names or with empty column names after cleaning
        bg_info = bg_info[
            bg_info["column_name"].notna() & (bg_info["column_name"] != "")
        ]

        if not bg_info.empty:
            # Create pivot table for background variables
            bg_info_pivot = bg_info.pivot_table(
                index="subject_id",
                columns="column_name",
                values="value",
                aggfunc="first",
            ).reset_index()

            # Merge with patient_info
            patient_info = pd.merge(
                patient_info, bg_info_pivot, on="subject_id", how="left"
            )

    return patient_info


def sort_features(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Sorting all concepts by 'subject_id' and 'abspos' (and 'index' and 'order' if they exist).
    """
    if "index" in concepts.columns and "order" in concepts.columns:
        concepts = concepts.sort_values(
            ["subject_id", "abspos", "index", "order"]
        )  # could maybe be done more optimally, is a bit slow
        concepts = concepts.drop(columns=["index", "order"])
    else:
        concepts = concepts.sort_values(["subject_id", "abspos"])

    return concepts


def create_segments(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign segments to the concepts DataFrame based on 'ADMISSION_ID', ensuring that
    events are ordered correctly within each 'PID'.
    Parameters:
        concepts: concepts with 'PID', 'ADMISSION_ID', and 'abspos' columns.
    Returns:
        concepts with a new 'segment' column
    """
    concepts = _assign_admission_ids(concepts)
    concepts["segment"] = np.nan

    # Assign maximum segment to 'Death' concepts
    concepts["segment"] = concepts.groupby("subject_id")["admission_id"].transform(
        normalize_segments_series
    )
    concepts = _assign_segments(concepts)
    concepts = assign_segments_to_death(concepts)
    return concepts


def normalize_segments_series(series: pd.Series) -> pd.Series:
    # Convert to string to ensure consistent types and avoid warnings
    series = series.astype(str)
    return series.factorize(use_na_sentinel=False)[0]


def _assign_segments(df):
    """
    Assign segments to the concepts DataFrame based on 'admission_id'
    """
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df["segment"] = df.groupby("subject_id")["admission_id"].transform(
        normalize_segments_series
    )
    return df


def assign_segments_to_death(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the maximum segment to 'DOD' concepts within each 'subject_id'.
    Parameters:
        df with 'subject_id', 'code', and 'segment' columns.
    Returns:
        df with 'DOD' concepts assigned to the maximum segment.
    """
    # Compute the maximum segment per 'subject_id'
    max_segment = (
        df.groupby("subject_id")["segment"].max().rename("max_segment").reset_index()
    )
    # Merge and assign
    df = df.merge(max_segment, on="subject_id", how="left")
    df["segment"] = df["segment"].where(df["code"] != "DOD", df["max_segment"])
    return df.drop(columns=["max_segment"])


def _assign_admission_ids(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'admission_id' to each row in concepts based on 'ADMISSION' and 'DISCHARGE' events.
    Assigns the same 'admission_id' to all events between 'ADMISSION' and 'DISCHARGE' events.
    If no 'ADMISSION' and 'DISCHARGE' events are present, assigns a new 'admission_id' to all events
    if the time between them is greater than 48 hours.
    """

    def _get_adm_id():
        return str(uuid.uuid4())

    # Work with a copy to avoid modifying the original
    result = concepts.copy()
    result["admission_id"] = None
    result["admission_id"] = result["admission_id"].astype(object)

    result = result.sort_values(by=["subject_id", "time"])

    has_admission = (result["code"].str.startswith("ADM_ADMISSION", na=False)).any()
    has_discharge = (result["code"].str.startswith("ADM_DISCHARGE", na=False)).any()

    if has_admission and has_discharge:
        new_result = _assign_explicit_admission_ids(result, _get_adm_id)
    else:
        new_result = _assign_time_based_admission_ids(result, _get_adm_id)

    result["admission_id"] = new_result["admission_id"]

    return result


def _assign_explicit_admission_ids(
    patient_data: pd.DataFrame, get_adm_id_func
) -> pd.DataFrame:
    """
    Assign admission IDs based on explicit ADMISSION and DISCHARGE events.
    Events outside admission periods are grouped by 48-hour rule.
    Admission IDs are only shared between events of the same patient.
    """
    patient_data = patient_data.copy()

    if len(patient_data) == 0:
        patient_data["admission_id"] = None
        patient_data["admission_id"] = patient_data["admission_id"].astype(object)
        return patient_data

    # Pre-process codes and timestamps to avoid repeated lookups
    codes = patient_data["code"].fillna("").values
    timestamps = patient_data["time"].values
    pids = patient_data["subject_id"].values

    # Initialize result array
    admission_ids = [None] * len(patient_data)

    # Track admission state per patient
    patient_states = {}  # pid -> (current_admission_id, current_outside_id, last_timestamp)

    # Process events using direct array iteration instead of iterrows
    for i, (code, timestamp, pid) in enumerate(zip(codes, timestamps, pids)):
        # Initialize patient state if not exists
        if pid not in patient_states:
            patient_states[pid] = (None, None, None)

        current_admission_id, current_outside_id, last_timestamp = patient_states[pid]

        if code.startswith("ADM_ADMISSION"):
            # Start new admission
            current_admission_id = get_adm_id_func()
            admission_ids[i] = current_admission_id
            # Reset outside admission tracking
            current_outside_id = None
            last_timestamp = None

        elif code.startswith("ADM_ADMISSION"):
            # End current admission
            if current_admission_id is not None:
                admission_ids[i] = current_admission_id
            else:
                # Discharge without admission - assign unique ID
                admission_ids[i] = get_adm_id_func()
            current_admission_id = None
            # Reset outside admission tracking
            current_outside_id = None
            last_timestamp = None

        else:
            # Regular event
            if current_admission_id is not None:
                # Inside admission period
                admission_ids[i] = current_admission_id
            else:
                # Outside admission period - apply 48-hour rule
                if (
                    current_outside_id is None
                    or last_timestamp is None
                    or pd.isna(timestamp)
                    or pd.isna(last_timestamp)
                    or (
                        pd.Timestamp(timestamp) - pd.Timestamp(last_timestamp)
                    ).total_seconds()
                    > 48 * 3600
                ):
                    # Start new outside-admission group
                    current_outside_id = get_adm_id_func()

                admission_ids[i] = current_outside_id
                last_timestamp = timestamp

        # Update patient state
        patient_states[pid] = (current_admission_id, current_outside_id, last_timestamp)

    # Assign results using vectorized assignment
    patient_data["admission_id"] = pd.Series(
        admission_ids, index=patient_data.index, dtype=object
    )

    return patient_data


def _assign_time_based_admission_ids(
    patient_data: pd.DataFrame, get_adm_id_func
) -> pd.DataFrame:
    """
    Assign admission IDs based on 48-hour time gaps.
    Admission IDs are only shared between events of the same patient.
    """
    patient_data = patient_data.copy()

    if len(patient_data) == 0:
        patient_data["admission_id"] = None
        patient_data["admission_id"] = patient_data["admission_id"].astype(object)
        return patient_data

    # Calculate time differences using vectorized operations
    time_diff = patient_data["time"].diff().dt.total_seconds()

    # Mark new admissions (first event or gap > 48 hours)
    new_admission = (time_diff > 48 * 3600) | time_diff.isna()

    # Also mark new admission when patient ID changes
    new_admission = new_admission | (
        patient_data["subject_id"] != patient_data["subject_id"].shift()
    )

    # Create admission groups using cumsum
    admission_groups = new_admission.cumsum()

    # Generate unique admission IDs for each group
    unique_groups = admission_groups.unique()
    group_to_id = {group: get_adm_id_func() for group in unique_groups}
    patient_data["admission_id"] = admission_groups.map(group_to_id).astype(object)

    return patient_data
