import bonsai
import os


def get_config_path():
    base_path = bonsai.__path__[0]
    return os.path.join(base_path, "configs")
