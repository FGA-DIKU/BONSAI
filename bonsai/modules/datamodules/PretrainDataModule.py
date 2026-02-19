import lightning as L
from typing import Literal
from torch.utils.data import DataLoader
from bonsai.functional.collate import dynamic_padding
from bonsai.modules.datasets.PretrainDataset import PretrainDataset


class PretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        vocabulary: list,
        # train_transforms: Optional[Compose] = pretrain_CPU_train_transforms,
        # val_transforms: Optional[Compose] = pretrain_CPU_val_transforms,
        # num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.vocabulary = vocabulary
        # self.train_transforms = train_transforms
        # self.val_transforms = val_transforms

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        self.train_dataset = PretrainDataset(
            self.train_split,
            vocabulary=self.vocabulary,
        )
        self.val_dataset = PretrainDataset(
            self.val_split,
            vocabulary=self.vocabulary,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,  # Why is shuffle false?
            collate_fn=dynamic_padding,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,
            collate_fn=dynamic_padding,
        )


if __name__ == "__main__":
    import torch
    import bonsai
    import os

    base_path = os.path.split(bonsai.__path__[0])[0]
    patients = torch.load(
        os.path.join(base_path, "outputs/pretraining/processed_data/patients_train.pt"),
        weights_only=False,
    )
    patientlist = []
    for patient in patients:
        sample = {
            "pid": torch.tensor(patient.pid, dtype=torch.long),
            "concept": torch.tensor(patient.concepts, dtype=torch.long),
            "abspos": torch.tensor(patient.abspos, dtype=torch.float),
            "segment": torch.tensor(patient.segments, dtype=torch.long),
            "age": torch.tensor(patient.ages, dtype=torch.half),
        }
        patientlist.append(sample)

    torch.save(
        patientlist,
        os.path.join(
            base_path, "outputs/pretraining/processed_data/DICTpatients_train.pt"
        ),
    )

    vocab = torch.load(
        os.path.join(base_path, "outputs/pretraining/processed_data/vocabulary.pt")
    )
    train_data = torch.load(
        os.path.join(
            base_path, "outputs/pretraining/processed_data/DICTpatients_train.pt"
        )
    )
    val_data = torch.load(
        os.path.join(
            base_path, "outputs/pretraining/processed_data/DICTpatients_train.pt"
        )
    )
    dm = PretrainDataModule(
        batch_size=2,
        num_workers=1,
        train_split=train_data,
        val_split=val_data,
        vocabulary=vocab,
    )

    dm.setup("fit")
    train_dl = dm.train_dataloader()
    train_iter = iter(train_dl)
    for i in range(5):
        print(next(train_iter))
