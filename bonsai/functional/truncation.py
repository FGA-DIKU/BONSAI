import numpy as np
import torch


def truncate_patient(patient, max_len: int, background_tokens_per_patient, sep_token=2):
    total_length = len(patient["code"])
    if total_length <= max_len:
        return patient

    # Determine how many items from the end we can keep
    tail_length = max_len - background_tokens_per_patient

    # If the boundary element is the SEP token, shift tail_length by 1
    if tail_length > 0 and patient["code"][-tail_length] == sep_token:
        tail_length = max(tail_length - 1, 0)

    for key, val in patient.items():
        if key not in ["target", "subject_id"]:
            patient[key] = torch.cat(
                (val[:background_tokens_per_patient], val[-tail_length:])
            )
    return patient


def truncate_subject(subject: dict, max_len: int, background_length: int) -> dict:
    if len(subject["code"]) > max_len:
        tokens_right = max_len - background_length
        tail = len(subject["code"]) - tokens_right

        # equivalent to torch.cat(feat[:background_length], feat[start:])
        idxs = np.r_[0:background_length, tail : len(subject["code"])]

        for embed_name in ["code", "abspos", "segment", "age"]:
            subject[embed_name] = subject[embed_name][idxs]
    return subject