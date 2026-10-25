from typing import Dict, List

import torch
from bonsai.modules.datasets.BaseDataset import BaseDataset

from bonsai.functional.censoring import censor_subject
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.truncation import truncate_subject


class FinetuneDataset(BaseDataset):
    def __init__(
        self,
        subjects: List[Dict],
        outcomes: Dict[int, dict],
        predict_token_id: int,
        background_length: int,
        max_len: int,
    ):
        super().__init__(subjects)
        self.outcomes = outcomes
        self.predict_token_id = predict_token_id
        self.background_length = background_length
        self.max_len = max_len

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        subject_outcome = self.outcomes[subject["subject_id"]]

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
        subject["attention_mask"] = torch.ones(len(subject["code"]), dtype=torch.bool)

        return subject
