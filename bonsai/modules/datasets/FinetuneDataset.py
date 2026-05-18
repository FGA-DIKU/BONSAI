from typing import List, Dict
import torch
from torch.utils.data import Dataset
from bonsai.functional.censoring import censor_subject
from bonsai.functional.truncation import truncate_subject
from bonsai.functional.normalization import normalize_segments
from copy import deepcopy


class FinetuneDataset(Dataset):
    def __init__(
        self,
        subjects: List[Dict],
        outcome_rows: List[dict],
        predict_token_id: int,
        background_length: int,
        max_len: int,
    ):
        if len(subjects) != len(outcome_rows):
            raise ValueError("subjects and outcome_rows must have the same length")
        self.subjects = subjects
        self.outcome_rows = outcome_rows
        self.predict_token_id = predict_token_id
        self.background_length = background_length
        self.max_len = max_len

    def __getitem__(self, index: int) -> dict:
        subject = deepcopy(self.subjects[index])
        subject_outcome = self.outcome_rows[index]

        subject["target"] = torch.tensor([subject_outcome["label"]], dtype=torch.long)

        if subject_outcome["censor_abspos"] is not None:
            subject = censor_subject(
                subject,
                censor_date_abspos=subject_outcome["censor_abspos"],
                predict_token_id=self.predict_token_id,
            )
        subject = truncate_subject(
            subject, max_len=self.max_len, background_length=self.background_length
        )

        subject["segment"] = normalize_segments(subject["segment"])
        subject["attention_mask"] = torch.ones(len(subject["code"]), dtype=torch.long)

        return subject

    def __len__(self):
        return len(self.subjects)
