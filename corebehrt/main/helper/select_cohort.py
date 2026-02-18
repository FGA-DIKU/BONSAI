from typing import List, Tuple

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.features.split import split_test
from corebehrt.functional.preparation.filter import select_first_event
from corebehrt.modules.cohort_handling.index_dates import IndexDateHandler
from corebehrt.modules.cohort_handling.patient_filter import (
    exclude_pids_from_df,
    filter_by_age,
    filter_by_categories,
    filter_by_death,
    filter_by_prior_outcomes,
    filter_df_by_pids,
)
from corebehrt.modules.features.loader import ConceptLoader


def select_cohort(
    path_cfg, selection_cfg, index_date_cfg, test_ratio, logger
) -> Tuple[List[str], pd.Series, List[str], List[str]]:
    """
    Select cohort by applying multiple filtering steps.

    The process includes:
      1. Loading and initial filtering of patient data.
      2. Determining index dates based on the selected mode (absolute or relative).
      3. Splitting the cohort into training/validation and test sets.
      4. Applying a test shift (for absolute index dates) to simulate future prediction,
         so that test patients have a shifted index date.
      5. Further filtering based on age, death, and prior outcomes.

    The rationale behind absolute index dates and test shift is to simulate a real-world deployment scenario:
      - The model is trained using data up to a fixed cutoff (absolute index date),
        with input data defined relative to that date.
      - For testing, the index date is shifted (using test_shift_hours) to mimic a
        future prediction scenario, ensuring that both input and outcome windows are
        appropriately aligned with a prospective evaluation.

    Args:
        path_cfg: Configuration dictionary
        selection_cfg: Configuration dictionary
        index_date_cfg: Configuration dictionary
        logger: Logger object
    Returns:
        Tuple of:
          - Final patient IDs (list)
          - Series of index dates (with potential test shift applied)
          - Train/validation patient IDs (list)
          - Test patient IDs (list)
    """

    logger.info("Loading data")
    patients_info, outcomes, exposures, initial_pids, exclude_pids = load_data(path_cfg)

    # Remove duplicate patient records (keep first occurrence)
    patients_info = patients_info.drop_duplicates(subset=PID_COL, keep="first")
    logger.info("N patients_info: %d", len(patients_info))
    logger.info("Patients in initial_pids: %d", len(initial_pids))

    if initial_pids:
        logger.info("Filtering by initial_pids")
        patients_info = filter_df_by_pids(patients_info, initial_pids)
        log_patient_num(logger, patients_info)

    if exclude_pids:
        logger.info("Filtering by exclude_pids")
        patients_info = exclude_pids_from_df(patients_info, exclude_pids)
        log_patient_num(logger, patients_info)

    if selection_cfg.get("exposed_only", False):
        logger.info("Filtering by exposures")

        patients_info = filter_df_by_pids(patients_info, exposures[PID_COL])
        log_patient_num(logger, patients_info)

    if selection_cfg.get("categories", False):
        logger.info("Filtering by categories")
        patients_info = filter_by_categories(patients_info, selection_cfg.categories)
        log_patient_num(logger, patients_info)

    # Determine index dates for all patients
    # For absolute mode, a fixed date is assigned; for relative, it's computed based on exposures.
    logger.info("Determining index dates")
    print(len(patients_info), len(exposures))
    mode = index_date_cfg["mode"]
    index_dates = IndexDateHandler.determine_index_dates(
        patients_info,
        mode,
        absolute_timestamp=index_date_cfg[mode].get("date"),
        n_hours_from_exposure=index_date_cfg[mode].get("n_hours_from_exposure"),
        exposures=exposures,
    )

    # This split is done after index date calculation but before any filtering based on index dates
    # the final split ratios might be slightly different from the test_ratio due to the filtering steps
    train_val_pids, test_pids = split_test(patients_info[PID_COL].tolist(), test_ratio)

    # For out-of-time evaluation with absolute index dates:
    # If a test shift is provided, adjust the index dates for test patients.
    # This simulates that while the model is trained on data up to the cutoff date,
    # predictions (and corresponding outcome follow-up) are made on a later time period.
    if mode == "absolute":
        test_shift_hours = index_date_cfg["absolute"].get("test_shift_hours", 0)
        if test_shift_hours:
            index_dates.loc[test_pids] += pd.Timedelta(hours=test_shift_hours)

    # Merge the (possibly shifted) index dates back into the patient info DataFrame.
    # The TIMESTAMP_COL now holds the index date for each patient.
    patients_info = patients_info.merge(
        index_dates, on=PID_COL
    )  # the TIMESTAMP column is the index date.

    if selection_cfg.get("age", False):
        logger.info("Filtering by age")
        patients_info = filter_by_age(
            patients_info,
            min_age=selection_cfg.age.get("min_years"),
            max_age=selection_cfg.age.get("max_years"),
        )
        log_patient_num(logger, patients_info)

    logger.info("Applying additional exclusion filters")
    patients_info = filter_by_death(patients_info)
    log_patient_num(logger, patients_info)

    if selection_cfg.get("exclude_prior_outcomes", False):
        logger.info("Filtering by prior outcomes")
        patients_info = filter_by_prior_outcomes(patients_info, outcomes)
        log_patient_num(logger, patients_info)
    return (
        patients_info[PID_COL].unique().tolist(),
        index_dates,
        train_val_pids,
        test_pids,
    )


def log_patient_num(logger, patients_info):
    logger.info(f"Patient number: {len(patients_info[PID_COL].unique())}")


def load_data(
    path_cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Load patient, outcomes, and exposures data."""
    patients_info = ConceptLoader.read_file(path_cfg.patients_info)
    outcomes = ConceptLoader.read_file(path_cfg.outcome)

    exposures = (
        ConceptLoader.read_file(path_cfg.exposure)
        if path_cfg.get("exposure", False)
        else outcomes
    )

    exposures = select_first_event(exposures, PID_COL, TIMESTAMP_COL)

    initial_pids = (
        torch.load(path_cfg.initial_pids, weights_only=False) if path_cfg.get("initial_pids", False) else []
    )

    exclude_pids = (
        torch.load(path_cfg.exclude_pids, weights_only=False) if path_cfg.get("exclude_pids", False) else []
    )

    return patients_info, outcomes, exposures, initial_pids, exclude_pids
