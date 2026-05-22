from hydra.core.hydra_config import HydraConfig


def get_experiment_output_path():
    return HydraConfig.get().runtime.output_dir
