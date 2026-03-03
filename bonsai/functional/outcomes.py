from typing import List, Literal, Optional, Dict
from datetime import datetime
import pandas as pd


def find(df, conditions: List, dependence: Literal["independent", "dependent"]):
    """Returns the first row (priority based on condition order) for each patient that matches the conditions"""
    # Initialization
    df["_prio"] = pd.Series()
    masks = False
    subject_sets = []

    for i, cond in enumerate(conditions):
        cond_mask = df[cond["col"]].isin(cond["vals"])  # Rows that meet condition
        masks |= cond_mask  # OR operation
        df["_prio"] = df["_prio"].mask(
            cond_mask, i
        )  # Set priority (to take first row later)
        subject_sets.append(
            set(df.loc[cond_mask, "subject_id"])
        )  # Get subject that match condition

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
    res = df[df["subject_id"].isin(matched_subjects) & masks]

    # Take first row based on `conditions` ordering
    res = (
        res.sort_values("_prio")
        .groupby("subject_id", sort=False, as_index=False)
        .first()
    )
    res = res.drop(columns="_prio")
    return res


def set_dates(
    date_type: Literal["relative", "absolute"],
    outcome_dates: Optional[pd.Series] = None,
    hour_shift: Optional[int] = None,
    date: Optional[Dict[str, int]] = None,
):
    if date_type == "absolute":
        assert date is not None
        return datetime(**date)
    elif date_type == "relative":
        assert outcome_dates is not None
        assert hour_shift is not None
        return outcome_dates + pd.Timedelta(hours=hour_shift)
    else:
        raise ValueError(
            f"Date_type only allowed to be [relative, absolute], not {date_type}"
        )
