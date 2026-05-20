import numpy as np
import os
from bonsai.paths import get_models_path
from typing import Union


def generate_unused_run_id(
    model_dir: str = get_models_path(), string_match="run_id="
) -> Union[None, int]:
    global_used_ids = [0]
    for dirpath, _, _ in os.walk(model_dir):
        d = os.path.split(dirpath)[1]
        if d.startswith(string_match):
            global_used_ids.append(int(d.replace(string_match, "")))

    # find a globally unique one
    run_id = 0
    while run_id in global_used_ids:
        run_id = int(np.random.randint(1000, 999999))

    return run_id
