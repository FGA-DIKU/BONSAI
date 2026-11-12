from pathlib import Path

from hydra.core.hydra_config import HydraConfig


def get_experiment_output_path():
    return HydraConfig.get().runtime.output_dir


def check_run_dir_unused(run_dir, ckpt_path) -> None:
    """Refuse to start a fresh run in a folder that already holds checkpoints."""
    if ckpt_path is None and any(Path(run_dir).glob("*.ckpt")):
        raise FileExistsError(
            f"{run_dir} already has checkpoints. Pass paths.ckpt_path to resume, "
            "or leave out run_id to start a new run."
        )
