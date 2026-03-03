import numpy as np
import torch


def truncate_patient(patient, max_len: int, background_tokens_per_patient, sep_token=2):
    total_length = len(patient["codes"])
    if total_length <= max_len:
        return patient

    # Determine how many items from the end we can keep
    tail_length = max_len - background_tokens_per_patient

    # If the boundary element is the SEP token, shift tail_length by 1
    if tail_length > 0 and patient["codes"][-tail_length] == sep_token:
        tail_length = max(tail_length - 1, 0)

    for key, val in patient.items():
        if key not in ["target", "subject_id"]:
            patient[key] = torch.cat(
                (val[:background_tokens_per_patient], val[-tail_length:])
            )
    return patient


def truncate_subject(subject: dict, max_len: int) -> dict:
    if len(subject["codes"]) <= max_len:
        return subject
    else:
        background_length = (subject["segments"] == 0).sum()
        tokens_right = max_len - background_length
        start = len(subject["codes"]) - tokens_right

        idxs = np.r_[0:background_length, start : len(subject["codes"])]
        return {
            "subject_id": subject["subject_id"],
            "codes": subject["codes"][idxs],
            "abspos": subject["abspos"][idxs],
            "segments": subject["segments"][idxs],
            "ages": subject["ages"][idxs],
        }
