import hydra
from omegaconf import DictConfig
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.paths import get_config_path
from dotenv import load_dotenv
from hydra.utils import get_class
import lightning as L

load_dotenv()


@hydra.main(
    config_path=get_config_path(),  # TODO: make this more flexible to allow for different config paths
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    data_module = PretrainDataModule(
        dataset_class=get_class(cfg.data.dataset_class),
        path_tokenized=cfg.data.path_tokenized,
        path_vocab=cfg.data.path_vocab,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        max_len=cfg.training.max_len,
        cutoff_date=cfg.training.cutoff_date,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")


if __name__ == "__main__":
    main()
