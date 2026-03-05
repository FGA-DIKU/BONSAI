import numpy as np

def truncate_subject(subject: dict, max_len: int, background_length: int) -> dict:
    if len(subject["code"]) > max_len:
        tokens_right = max_len - background_length
        tail = len(subject["code"]) - tokens_right

        # equivalent to torch.cat(feat[:background_length], feat[start:])
        idxs = np.r_[0:background_length, tail : len(subject["code"])]

        for embed_name in ["code", "abspos", "segment", "age"]:
            subject[embed_name] = subject[embed_name][idxs]
    return subject