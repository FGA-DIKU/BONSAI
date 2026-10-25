import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, subjects):
        lengths = torch.tensor(
            [len(s["code"]) for s in subjects],
            dtype=torch.long
        )

        self.offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                lengths.cumsum(0)
            ]
        )

        self.subject_ids = torch.tensor(
            [s["subject_id"] for s in subjects],
            dtype=torch.long
        )

        self.features = {
            key: torch.cat([s[key] for s in subjects])
            for key in subjects[0]
            if key != "subject_id"
        }

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])

        subject = {"subject_id": int(self.subject_ids[index])}

        for key, values in self.features.items():
            subject[key] = values[start:end].clone()

        return subject
