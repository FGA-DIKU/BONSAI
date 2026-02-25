import hydra
from omegaconf import DictConfig, OmegaConf
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.paths import get_config_path
import logging


@hydra.main(
    config_path=get_config_path(),  # TODO: make this more flexible to allow for different config paths
    # config_path="bonsai/configs",
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    pretrain_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = logging.getLogger("pretrain")

    data_module = PretrainDataModule(
        logger=logger,
        **pretrain_cfg["data"],
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

if __name__ == "__main__":
    main()
