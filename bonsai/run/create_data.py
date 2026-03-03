import logging
import hydra
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
import torch

from bonsai.paths import get_config_path
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from hydra.core.plugins import Plugins

from bonsai.modules.create_data import process_split
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer
from bonsai.functional.subject_data import prepare_subject_data

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="example_data",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    path_input_dir = Path(cfg.data.input_dir)
    path_output_dir = Path(cfg.data.output_dir)

    # Initialize tokenizer and vocabulary
    if cfg.tokenizer.vocabulary is not None:
        vocab = torch.load(path_input_dir / cfg.tokenizer.vocabulary)
    else:
        vocab = None
        assert cfg.splits[0] == "train", (
            "First split must be 'train' to build vocabulary before tokenizing other splits"
        )
    tokenizer = EHRTokenizer(
        vocabulary=vocab,
        cutoffs=cfg.tokenizer.cutoffs,
        sep_tokens=cfg.tokenizer.sep_tokens,
    )

    logging.info("create_data:")
    for split in cfg.splits:
        logging.info(f"process_split: {split}")
        process_split(
            split=split,
            path_input_dir=path_input_dir,
            path_output_dir=path_output_dir,
            tokenizer=tokenizer,
            exclude_regex=cfg.exclude_regex,
        )
        tokenizer.freeze_vocabulary()  # freeze after first split (train) to prevent data leakage

        # TODO: Should be moved to process_split going forward
        logging.info(f"prepare_subject_data: {split}")
        subject_data = prepare_subject_data(
            split_path=path_output_dir / split,
        )
        torch.save(subject_data, path_output_dir / f"subject_data_{split}.pt")

    torch.save(tokenizer.vocabulary, path_output_dir / "vocabulary.pt")


if __name__ == "__main__":
    main()
