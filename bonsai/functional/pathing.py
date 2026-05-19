from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def get_experiment_output_path():
    return HydraConfig.get().runtime.output_dir


def save_run_config(cfg: DictConfig, save_dir: str | Path) -> Path:
    """Write the resolved Hydra config next to checkpoints / metrics."""
    path = Path(save_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path
