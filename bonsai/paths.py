import os

from dotenv import load_dotenv

load_dotenv()


def get_config_path():
    return os.getenv("BONSAI_CONFIG_PATH")


def get_data_path():
    return os.getenv("BONSAI_PROCESSED_DATA")


def get_models_path():
    return os.getenv("BONSAI_MODELS")
