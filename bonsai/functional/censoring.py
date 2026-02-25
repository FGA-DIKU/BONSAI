import pandas as pd
from bisect import bisect_right
from typing import Dict


def censor_patient(
    patient: Dict,
    censor_date: pd.Series,
    predict_token_id: int,
    concept_id_to_delay: dict = None,
    sep_token: int = 2,
):
    if concept_id_to_delay is None:
        patient = censor_patient_without_delays(
            patient=patient,
            censor_date=censor_date,
            predict_token_id=predict_token_id,
            sep_token=sep_token,
        )
    else:
        patient = censor_patient_with_delays(
            patient=patient,
            censor_date=censor_date,
            predict_token_id=predict_token_id,
            concept_id_to_delay=concept_id_to_delay,
        )
    return patient


def censor_patient_without_delays(
    patient: Dict,
    censor_date: pd.Series,
    predict_token_id: int,
    sep_token: int,
) -> Dict:
    """
    Censors a patient's data by truncating all attributes at the censor date,
    then appends a CLS token with the censoring information.

    The function shortens the concept, abspos, segments, and ages lists of a Dict object so that only entries occurring before or at the patient's censor date are retained, then adds a predict token at the end.
    """
    # Find the position where censor_date fits in the sorted abspos list
    idx = bisect_right(patient["abspos"], censor_date)

    if patient["concept"][:idx][-1] == sep_token:
        idx -= 1

    # Slice everything up to idx
    patient["concepts"] = patient["concepts"][:idx]
    patient["abspos"] = patient["abspos"][:idx]
    patient["segments"] = patient["segments"][:idx]
    patient["ages"] = patient["ages"][:idx]

    patient = append_predict_token(patient, predict_token_id, censor_date)

    return patient


def censor_patient_with_delays(
    patient: Dict,
    censor_date: pd.Series,
    predict_token_id: int,
    concept_id_to_delay: dict = None,
) -> Dict:
    """
    Censors a patient's data using concept-specific delays applied to their censor date.
    Adds the predict token with age at censoring at the end of the sequence.

    For each concept in the patient's record, calculates an effective censor date by adding a delay (if specified) to the base censor date for the patient. Retains only those concepts and corresponding attributes whose timestamps are less than or equal to their effective censor dates.
    Then adds predict token with age at censoring as a final token.
    """
    # Initialize keep mask
    keep_mask = [False] * len(patient["concepts"])

    # Process each concept with its appropriate delay
    for i, (concept, abspos) in enumerate(zip(patient["concepts"], patient["abspos"])):
        # Get delay for this concept (0 for unmapped concepts)
        delay = concept_id_to_delay.get(concept, 0)

        # Calculate effective censor date for this concept
        effective_censor_date = censor_date + delay

        # Keep this concept if it's before or at the effective censor date
        if abspos <= effective_censor_date:
            keep_mask[i] = True

    # Apply the mask to all patient attributes
    patient["concepts"] = [c for i, c in enumerate(patient["concepts"]) if keep_mask[i]]
    patient["abspos"] = [a for i, a in enumerate(patient["abspos"]) if keep_mask[i]]
    patient["segments"] = [s for i, s in enumerate(patient["segments"]) if keep_mask[i]]
    patient["ages"] = [a for i, a in enumerate(patient["ages"]) if keep_mask[i]]

    patient = append_predict_token(patient, predict_token_id, censor_date)

    return patient


def append_predict_token(
    patient: Dict, predict_token_id: int, censor_date: float
) -> Dict:
    patient["concepts"].append(predict_token_id)
    patient["abspos"].append(float(censor_date))
    patient["segments"].append(
        patient["segments"][-1] + 1 if patient["segments"] else 0
    )  # Use 0 as default segment if segments list is empty
    age_in_years = float((censor_date - patient["abspos"][0]) / (365.25 * 24))
    patient["ages"].append(age_in_years)
    return patient


def cutoff_subject(subject: Dict, cutoff_date: float) -> Dict:
    """
    Cuts off a subject's data at the specified cutoff date by removing all concepts and corresponding attributes that occur after the cutoff date.
    """
    # Find the position where cutoff_date fits in the sorted abspos list
    idx = bisect_right(subject["abspos"], cutoff_date)

    # Slice everything up to idx
    subject["codes"] = subject["codes"][:idx]
    subject["abspos"] = subject["abspos"][:idx]
    subject["segments"] = subject["segments"][:idx]
    subject["ages"] = subject["ages"][:idx]

    return subject