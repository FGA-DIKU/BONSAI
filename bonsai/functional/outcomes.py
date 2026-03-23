from typing import List, Literal, Optional, Dict, Tuple
from datetime import datetime
import pandas as pd


def match(
    df: pd.DataFrame, conditions: List, dependence: Literal["independent", "dependent"]
) -> pd.DataFrame:
    """Returns the first row (priority based on condition order) for each subject that matches the conditions"""
    # Initialization
    df["_prio"] = pd.Series()
    row_masks = False
    subject_sets = []

    # Find matches (dataframe rows AND subject_ids) of conditions
    for i, cond in enumerate(conditions):
        cond_mask = df[cond["col"]].isin(cond["vals"])  # Rows that meet condition
        row_masks |= cond_mask  # OR operation
        df["_prio"] = df["_prio"].mask(
            cond_mask, i
        )  # Set priority (to take first row later)
        subject_sets.append(
            set(df.loc[cond_mask, "subject_id"])
        )  # Get subjects that match condition

    # Toggle betweens any or all conditions met
    if dependence == "independent":
        matched_subjects = set.union(*subject_sets)  # Any condition met
    elif dependence == "dependent":  # TODO: Implement time_window
        matched_subjects = set.intersection(*subject_sets)  # All conditions met
    else:
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    # Get matched subjects AND rows
    res = df[df["subject_id"].isin(matched_subjects) & row_masks]

    # Take first row based on `conditions` ordering
    res = (
        res.sort_values("_prio")
        .groupby("subject_id", sort=False, as_index=False)
        .first()
    )
    res = res.drop(columns="_prio")
    return res


def set_dates(
    date_type: Literal["relative", "absolute", "exposure"],
    relative_dates: Optional[pd.Series] = None,  # Required for relative
    relative_hour_shift: Optional[Dict[str, int]] = None,  # Required for relative
    absolute_date: Optional[Dict[str, int]] = None,  # Required for absolute
    subjects: Optional[pd.DataFrame] = None,  # Required for exposure
    df: Optional[pd.DataFrame] = None,  # Required for exposure
    exposure_dependence: Optional[
        Literal["independent", "dependent"]
    ] = None,  # Required for exposure
    exposure_conditions: Optional[list] = None,  # Required for exposure
):
    """Return dates based on three `date_type` options:

    Relative: Returns `relative_dates + relative_hour_shift`.
    Absolute: Returns `absolute_date` as a datetime object.
    Exposure: Returns the subjects matching `dependence` and `conditions` using the `match` function on `df` and merges with `subjects`.

    """
    if date_type == "absolute":
        assert absolute_date is not None
        return datetime(**absolute_date)
    elif date_type == "relative":
        assert relative_dates is not None
        assert relative_hour_shift is not None
        return relative_dates + pd.Timedelta(hours=relative_hour_shift)
    elif date_type == "exposure":
        assert subjects is not None
        assert df is not None
        assert exposure_dependence is not None
        assert exposure_conditions is not None
        result = match(
            df, conditions=exposure_conditions, dependence=exposure_dependence
        )
        return subjects.merge(
            result[["subject_id", "time"]], on="subject_id", how="left"
        )["time"]
    else:
        raise ValueError(
            f"Date_type only allowed to be [relative, absolute, exposure], not {date_type}"
        )


def binarize_outcomes(
    outcomes: pd.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
) -> Dict[int, dict]:
    time_delta_datetime = outcomes["outcome_date"] - outcomes["index_date"]
    time_delta_hours = time_delta_datetime.dt.days * 24

    outcomes_in_prediction_window = n_hours_start_include <= time_delta_hours
    if n_hours_end_include is not None:
        outcomes_in_prediction_window &= time_delta_hours <= n_hours_end_include

    outcomes["label"] = outcomes_in_prediction_window.astype(int)
    return (
        outcomes[["subject_id", "label", "censor_abspos"]]
        .set_index("subject_id")
        .to_dict(orient="index")
    )


def split_and_binarize_outcomes(
    outcomes,
    train_key: str,
    val_key: str,
    test_key: str,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, dict]]:
    train_outcomes = outcomes[outcomes["split"] == train_key]
    train_outcomes = binarize_outcomes(
        train_outcomes, n_hours_start_include, n_hours_end_include
    )
    val_outcomes = outcomes[outcomes["split"] == val_key]
    val_outcomes = binarize_outcomes(
        val_outcomes, n_hours_start_include, n_hours_end_include
    )
    test_outcomes = outcomes[outcomes["split"] == test_key]
    test_outcomes = binarize_outcomes(
        test_outcomes, n_hours_start_include, n_hours_end_include
    )

    return train_outcomes, val_outcomes, test_outcomes
