import torch

from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset


class TTEDataset(FinetuneDataset):
    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        subject["duration"] = torch.tensor(
            [self.outcomes[subject["subject_id"]]["duration"]], dtype=torch.float
        )
        return subject
