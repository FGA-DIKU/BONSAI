from bonsai.modules.transforms.masking import ConceptMasker
from torch.utils.data import Dataset
from typing import List, Dict


class PretrainDataset(Dataset):
    def __init__(
        self,
        patients: List[Dict],
        vocabulary: Dict,
        masking_select_ratio: float = 1.0,
        masking_ratio: float = 0.8,
        masking_replace_ratio: float = 0.1,
        masking_ignore_special_tokens: bool = True,
    ):
        self.patients = patients
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(
            vocabulary,
            masking_select_ratio,
            masking_ratio,
            masking_replace_ratio,
            masking_ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        patient = self.patients[index]
        masked_concepts, target, attention_mask = self.masker.mask_patient_concepts(
            patient["concept"]
        )
        patient["concept"] = masked_concepts
        patient["target"] = target
        patient["attention_mask"] = attention_mask
        return patient

    def __len__(self):
        return len(self.patients)
