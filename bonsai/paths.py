import bonsai
import os


# def get_config_path():
#    base_path = bonsai.__path__[0]
#    return base_path.replace("bonsai", "configs")


def get_config_path():
    return os.getenv("BONSAI_CONFIG_PATH")


def get_data_path():
    return os.getenv("BONSAI_PROCESSED_DATA")


def get_models_path():
    return os.getenv("BONSAI_MODELS")
