from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from bonsai.functional.censoring import censor_subject
from bonsai.functional.features import compute_abspos
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.truncation import truncate_subject


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
        subject = deepcopy(self.subjects[index])
        if self.cutoff_date is not None:
            subject = censor_subject(subject, self.cutoff_date)
        truncated_subject = truncate_subject(
            subject, self.max_len, self.background_length
        )
        truncated_subject["attention_mask"] = torch.ones(
            len(truncated_subject["code"]), dtype=torch.bool
        )
        truncated_subject["segment"] = normalize_segments(truncated_subject["segment"])
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
        masking_mask_ratio: float = 0.8,
        masking_random_ratio: float = 0.1,
        masking_ignore_special_tokens: bool = True,
        cutoff_date: Optional[dict] = None,
    ):
        super().__init__(subjects, max_len, background_length, cutoff_date=cutoff_date)
        self.vocabulary = vocabulary

        self.masking_select_ratio = masking_select_ratio
        self.masking_mask_ratio = masking_mask_ratio
        self.masking_random_ratio = masking_random_ratio
        self.masking_n_special_tokens = (
            len([token for token in vocabulary if token.startswith("[")])
            if masking_ignore_special_tokens
            else 0
        )

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        numeric_values = subject.get("numeric_value")
        masked_codes, masked_numeric_values, target, numeric_target = (
            self.mask_patient_codes(subject["code"], numeric_values)
        )
        # Collate / model read "code"; mask_patient_codes returns the masked sequence.
        subject["code"] = masked_codes
        subject["target"] = target
        if numeric_target is not None:
            subject["numeric_value"] = masked_numeric_values
            subject["numeric_target"] = numeric_target
        return subject

    def mask_patient_codes(
        self,
        codes: torch.Tensor,
        numeric_values: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        target = codes.clone()
        probability_vector = torch.full(target.shape, self.masking_select_ratio)

        # Ignore special tokens
        special_token_mask = codes < self.masking_n_special_tokens
        probability_vector.masked_fill_(special_token_mask, value=0.0)

        # Get MLM mask
        selected_indices = torch.bernoulli(probability_vector).bool()
        target[~selected_indices] = -100

        # Replace with [MASK]
        indices_mask = (
            torch.bernoulli(torch.full(target.shape, self.masking_mask_ratio)).bool()
            & selected_indices
        )
        codes = codes.clone()
        codes[indices_mask] = self.vocabulary["[MASK]"]

        if numeric_values is not None:
            numeric_values = numeric_values.clone()
            numeric_target = numeric_values.clone()
            # Hide input values at [MASK]; keep targets only at those positions.
            numeric_values[indices_mask] = float("nan")
            numeric_target[~indices_mask] = float("nan")
        else:
            numeric_target = None

        # Replace with random word and Account for already masked tokens
        random_ratio = self.masking_random_ratio / (1 - self.masking_mask_ratio)
        indicies_random = (
            torch.bernoulli(torch.full(target.shape, random_ratio)).bool()
            & selected_indices
            & ~indices_mask
        )
        random_words = torch.randint(
            self.masking_n_special_tokens,
            len(self.vocabulary),
            target.shape,
            dtype=codes.dtype,
        )
        codes[indicies_random] = random_words[indicies_random]
        return codes, numeric_values, target, numeric_target


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
        subject["target"] = subject["target"].masked_fill(subject["target"] == 0, -100)

        if "numeric_value" in subject:
            values = subject["numeric_value"]
            subject["numeric_target"] = values[1:].clone()
            subject["numeric_value"] = values[:-1]

            subject["numeric_target"] = subject["numeric_target"].masked_fill(
                subject["target"] == -100, float("nan")
            )

        for key in ["code", "abspos", "segment", "age", "attention_mask"]:
            subject[key] = subject[key][:-1]
        return subject
