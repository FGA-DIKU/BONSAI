# prepared_data: ./outputs/pretraining/processed_data/
# PREPARED_TRAIN_PATIENTS = "patients_train.pt"
# PREPARED_VAL_PATIENTS = "patients_val.pt"
# PREPARED_ALL_PATIENTS = "patients.pt"
# %%
from typing import List, Literal, Optional
import torch

# splits = torch.load(
#    "/Users/zcr545/Desktop/Projects/repos/BONSAI/outputs/pretraining/processed_data/pids_train.pt"
# )
# patients = torch.load(
#    "/Users/zcr545/Desktop/Projects/repos/BONSAI/outputs/pretraining/processed_data/patients_train.pt",
#    weights_only=False,
# )
from torch.utils.data import DataLoader
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
import random
import copy
import os

CONFIG_PATH = "/Users/zcr545/Desktop/Projects/repos/BONSAI/corebehrt/configs/prepare_pretrain.yaml"
cfg = load_config(CONFIG_PATH)

os.chdir("/Users/zcr545/Desktop/Projects/repos/BONSAI")


def split_pids_into_train_val(data, val_split: float):
    """
    Splits data into train and val.
    Returns two PatientDatasets.
    """
    assert val_split < 1 and val_split > 0, "Split must be between 0 and 1"
    train_split = 1 - val_split
    random.seed(42)
    pids = copy.deepcopy(data.get_pids())
    random.shuffle(pids)
    train_pids = pids[: int(len(pids) * train_split)]
    val_pids = pids[int(len(pids) * train_split) :]
    train_data = data.filter_by_pids(train_pids)
    val_data = data.filter_by_pids(val_pids)
    return train_data, val_data


def save_pids_splits(train_data, val_data, save_dir: str) -> None:
    """
    Save train and val data to a folder.
    Assumes that the data has a column named PID.
    """
    os.makedirs(save_dir, exist_ok=True)
    train_pids = train_data.get_pids()
    val_pids = val_data.get_pids()
    torch.save(train_pids, os.path.join(save_dir, "pids_train.pt"))
    torch.save(val_pids, os.path.join(save_dir, "pids_val.pt"))


# Setup directories
# DirectoryPreparer(cfg).setup_prepare_pretrain()
data = DatasetPreparer(cfg).prepare_pretrain_data(save_data=False)


train_data, val_data = split_pids_into_train_val(data, cfg.data.get("val_ratio", 0.2))

save_pids_splits(train_data, val_data, cfg.paths.prepared_data)
train_data.save(cfg.paths.prepared_data, suffix="_train")
val_data.save(cfg.paths.prepared_data, suffix="_val")

# %%


from bonsai.functional.collate import dynamic_padding


class PretrainDataset(Dataset):
    def __init__(
        self,
        patients: List[PatientData],
        vocabulary: dict,
        select_ratio: float,
        masking_ratio: float = 0.8,
        replace_ratio: float = 0.1,
        ignore_special_tokens: bool = True,
    ):
        self.patients = patients
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(
            vocabulary,
            select_ratio,
            masking_ratio,
            replace_ratio,
            ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        """
        TODO: Save these items as tensors already and potentially as a dict instead of the DataClass.
        1. Retrieve the PatientData.
        2. Mask the 'concepts'.
        3. Convert everything to torch.Tensor.
        4. Return a dict that PyTorch can collate into a batch.
        """
        patient = self.patients[index]
        concepts = torch.tensor(patient.concepts, dtype=torch.long)
        masked_concepts, target = self.masker.mask_patient_concepts(concepts)
        attention_mask = torch.ones_like(masked_concepts)
        sample = {
            CONCEPT_FEAT: masked_concepts,
            TARGET: target,
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,
        }

        return sample

    def __len__(self):
        return len(self.patients)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        # train_transforms: Optional[Compose] = pretrain_CPU_train_transforms,
        # val_transforms: Optional[Compose] = pretrain_CPU_val_transforms,
        # num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
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
            self.train_split, transforms=self.train_transforms
        )
        self.val_dataset = PretrainDataset(
            self.val_split, transforms=self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,
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
        )

    # Get data
    train_data = PatientDataset(
        torch.load(
            join(cfg.paths.prepared_data, PREPARED_TRAIN_PATIENTS), weights_only=False
        )
    )
    val_data = PatientDataset(
        torch.load(
            join(cfg.paths.prepared_data, PREPARED_VAL_PATIENTS), weights_only=False
        )
    )
    vocab = load_vocabulary(cfg.paths.prepared_data)

    # Initialize datasets
    train_dataset = MLMDataset(train_data.patients, vocab, **cfg.data.dataset)
    val_dataset = MLMDataset(val_data.patients, vocab, **cfg.data.dataset)
