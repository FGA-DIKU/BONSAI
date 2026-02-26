import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.datamodules.base import BaseDataModule
from bonsai.paths import get_config_path
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from hydra.core.plugins import Plugins
from dotenv import load_dotenv

load_dotenv()

Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="create_data",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    create_data_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    data_module = BaseDataModule(**create_data_cfg)
    data_module.prepare_data()


if __name__ == "__main__":
    main()
