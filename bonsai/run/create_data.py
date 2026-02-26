import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from bonsai.paths import get_config_path
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from hydra.core.plugins import Plugins
from dotenv import load_dotenv
from pathlib import Path
from bonsai.functional.subject_data import prepare_subject_data
from bonsai.modules.create_data import create_features_and_tokenize
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer
import torch

load_dotenv()

Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="create_data",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    path_input_dir = Path(cfg.data.input_dir)
    path_output_dir = Path(cfg.data.output_dir)

    tokenizer = EHRTokenizer(
        vocabulary=None,
        sep_tokens=True,
    )

    assert cfg.splits[0] == "train", (
        "First split must be 'train' to build vocabulary before tokenizing other splits"
    )
    for split in cfg.splits:
        logging.info(f"prepare_data: {split}")
        create_features_and_tokenize(
            split=split,
            path_data=path_input_dir,
            path_tokenized=path_output_dir,
            tokenizer=tokenizer,
            exclude_regex=cfg.exclude_regex,
        )

        logging.info(f"prepare_subject_data: {split}")
        subject_data = prepare_subject_data(
            split_path=path_output_dir / split,
        )
        torch.save(subject_data, path_output_dir / f"subject_data_{split}.pt")
        tokenizer.freeze_vocabulary()  # freeze after first split (train) to prevent data leakage

    torch.save(tokenizer.vocabulary, path_output_dir / "vocabulary.pt")


if __name__ == "__main__":
    main()
