import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    # config_path=get_config_path(), # TODO: make this more flexible to allow for different config paths
    config_path="./corebehrt/configs",
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # TODO: implement path config for BONSAI
    # DirectoryPreparer(cfg).setup_pretrain()

    # TODO: implement logging for BONSAI
    # logger = logging.getLogger("pretrain")

    # TODO: Move path to the LightningTrainer so it auto-loads checkpoint from path
    # restart_path = cfg.paths.get("restart_model")
    # if restart_path:
    #    cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # TODO: Implement DataModule here and instantiate datasets in there.
    # train_data = PatientDataset(
    #    torch.load(join(cfg.paths.prepared_data, PREPARED_TRAIN_PATIENTS))
    # )
    # val_data = PatientDataset(
    #    torch.load(join(cfg.paths.prepared_data, PREPARED_VAL_PATIENTS))
    # )
    # vocab = load_vocabulary(cfg.paths.prepared_data)
    #     #train_dataset = MLMDataset(train_data.patients, vocab, **cfg.data.dataset)
    # val_dataset = MLMDataset(val_data.patients, vocab, **cfg.data.dataset)

    # TODO: Implement LightningModule here

    # TODO: Implement Lightning Trainer here and pass in model, optimizer, scheduler, datasets, etc.
