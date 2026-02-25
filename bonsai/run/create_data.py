import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.datamodules.base import BaseDataModule
from bonsai.paths import get_config_path


@hydra.main(
    config_path=get_config_path(),
    config_name="create_data",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    create_data_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = logging.getLogger("create_data")
    data_module = BaseDataModule(logger=logger, **create_data_cfg)
    data_module.prepare_data()
    # logger.info("Data preparation complete. Now running setup to load tokenized data into memory for training.")
    # data_module.setup()

if __name__ == "__main__":
    main()