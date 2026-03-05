from datetime import datetime
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from bonsai.functional.truncation import truncate_subject
from bonsai.functional.censoring import censor_subject
from bonsai.functional.features import compute_abspos
from bonsai.modules.transforms.masking import CodeMasker


class PretrainDataset(Dataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        background_length: int,
        cutoff_date: Optional[dict] = None,
    ):
        self.subjects = subjects
        self.max_len = max_len
        self.background_length = background_length
        self.cutoff_date = (
            compute_abspos(datetime(**cutoff_date)) if cutoff_date is not None else None
        )

    def __getitem__(self, index: int) -> dict:
        subject = self.subjects[index]
        if self.cutoff_date is not None:
            subject = censor_subject(subject, self.cutoff_date)
        truncated_subject = truncate_subject(
            subject, self.max_len, self.background_length
        )
        truncated_subject["attention_mask"] = torch.ones(
            len(truncated_subject["code"]), dtype=torch.long
        )
        return truncated_subject

    def __len__(self):
        return len(self.subjects)


class MLMPretrainDataset(PretrainDataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        background_length: int,
        vocabulary: Dict[str, int],
        masking_select_ratio: float,
        masking_ratio: float = 0.8,
        masking_random_ratio: float = 0.1,
        masking_ignore_special_tokens: bool = True,
        cutoff_date: Optional[dict] = None,
    ):
        super().__init__(subjects, max_len, background_length, cutoff_date=cutoff_date)
        self.vocabulary = vocabulary
        self.masker = CodeMasker(
            vocabulary,
            masking_select_ratio,
            masking_ratio,
            masking_random_ratio,
            masking_ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        masked_codes, target = self.masker.mask_patient_codes(subject["code"])
        subject["concept"] = masked_codes
        subject["target"] = target
        return subject


class ARPretrainDataset(PretrainDataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        background_length: int,
        cutoff_date: Optional[dict] = None,
    ):
        super().__init__(
            subjects, max_len + 1, background_length, cutoff_date=cutoff_date
        )  # +1 because we shift by one token in __getitem__

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        subject["target"] = subject["code"][1:]
        for key in ["code", "abspos", "segment", "age", "attention_mask"]:
            subject[key] = subject[key][:-1]
        return subject
