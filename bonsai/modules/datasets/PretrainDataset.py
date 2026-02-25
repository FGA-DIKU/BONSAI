from datetime import datetime
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from bonsai.functional.truncation import truncate_subject
from bonsai.functional.censoring import cutoff_subject
from bonsai.functional.features import compute_abspos
from bonsai.modules.transforms.masking import ConceptMasker


class PretrainDataset(Dataset):
    def __init__(self, subjects: List[Dict], max_len: int, cutoff_date: Optional[dict] = None):
        self.subjects = subjects
        self.max_len = max_len
        self.cutoff_date = compute_abspos(datetime(**cutoff_date)) if cutoff_date is not None else None

    def __getitem__(self, index: int) -> dict:
        subject = self.subjects[index]
        if self.cutoff_date is not None:
            subject = cutoff_subject(subject, self.cutoff_date)
        truncated_subject = truncate_subject(subject, self.max_len)
        return truncated_subject

    def __len__(self):
        return len(self.subjects)

class MLMPretrainDataset(PretrainDataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        vocabulary: Dict[str, int],
        masking_select_ratio: float,
        masking_ratio: float = 0.8,
        masking_random_ratio: float = 0.1,
        masking_ignore_special_tokens: bool = True,
        cutoff_date: Optional[dict] = None,
    ):
        super().__init__(subjects, max_len, cutoff_date=cutoff_date)
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(
            vocabulary,
            masking_select_ratio,
            masking_ratio,
            masking_random_ratio,
            masking_ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        masked_codes, target = self.masker.mask_patient_codes(
            subject["codes"]
        )
        subject["concept"] = masked_codes
        subject["target"] = target
        return subject
    
class ARPretrainDataset(PretrainDataset):
    def __init__(self,subjects: List[Dict], max_len: int, cutoff_date: Optional[dict] = None):
        super().__init__(subjects, max_len+1, cutoff_date=cutoff_date) # +1 because we shift by 1 in __getitem__

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        subject["target"] = subject["codes"][1:]
        for key in ["codes", "abspos", "segments", "ages"]:
            subject[key] = subject[key][:-1]
        return subject

