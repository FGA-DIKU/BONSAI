from typing import Optional, Dict, Tuple
import pandas as pd


def binarize_outcomes(
    outcomes: pd.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
) -> Dict[int, dict]:
    time_delta_datetime = outcomes["outcome_date"] - outcomes["index_date"]
    time_delta_hours = time_delta_datetime.dt.days * 24
    outcomes_in_prediction_window = time_delta_hours > n_hours_start_include
    if n_hours_end_include is not None:
        outcomes_in_prediction_window = time_delta_hours < n_hours_end_include
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
