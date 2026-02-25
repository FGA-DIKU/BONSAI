import bonsai
import os


def get_config_path():
    base_path = bonsai.__path__[0]
    return os.path.join(base_path, "configs")


def get_data_path():
    return os.getenv("BONSAI_PROCESSED_DATA")


def get_models_path():
    return os.getenv("BONSAI_MODELS")
