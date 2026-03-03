import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union
from bonsai.functional.censoring import censor_patient
from bonsai.functional.truncation import truncate_patient
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.features import compute_abspos


class FinetuneDataset(Dataset):
    # Outcomes are binary at this point
    # Censor times already take into account "n_hours_censoring"
    def __init__(
        self,
        patients: List[Dict],
        labels: pd.DataFrame,
        vocabulary: Dict,
        background_tokens_per_patient: int,
        max_len: int = 30,
        concept_id_to_delay: Optional[Union[str, None]] = None,
    ):
        self.patients = patients
        self.labels = labels
        self.vocabulary = vocabulary
        self.background_tokens_per_patient = background_tokens_per_patient
        self.max_len = max_len
        self.concept_id_to_delay = concept_id_to_delay

    def __getitem__(self, index: int) -> dict:
        patient = self.patients[index]

        patient_outcome = self.labels[
            self.labels["subject_id"] == patient["subject_id"].item()
        ]
        patient["target"] = torch.tensor(
            [patient_outcome["label"].item()], dtype=torch.long
        )
        if not pd.isnull(patient_outcome["censor_date"]).item():
            patient = censor_patient(
                patient=patient,
                censor_date_abspos=compute_abspos(
                    patient_outcome["censor_date"]
                ).item(),
                predict_token_id=self.vocabulary["[CLS]"],
                concept_id_to_delay=self.concept_id_to_delay,
            )

        patient = truncate_patient(
            patient=patient,
            max_len=self.max_len,
            background_tokens_per_patient=self.background_tokens_per_patient,
        )

        patient["segments"] = normalize_segments(patient["segments"])
        patient["attention_mask"] = torch.ones(len(patient["codes"]), dtype=torch.long)

        return patient

    def __len__(self):
        return len(self.patients)
