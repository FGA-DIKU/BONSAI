from bisect import bisect_right
from typing import Dict, Optional
import torch


def censor_subject(
    subject: Dict[str, torch.Tensor],
    censor_date_abspos: float,
    predict_token_id: Optional[int] = None,
) -> Dict:
    """
    Censors a subject's data by truncating all attributes at the censor date,
    OPTIONALLY: then appends a CLS token with the censoring information.
    """
    # Find the position where censor_date fits in the sorted abspos list
    idx = bisect_right(subject["abspos"].numpy(), censor_date_abspos)

    # Slice everything up to idx
    for embed_name in ["code", "abspos", "segment", "age"]:
        subject[embed_name] = subject[embed_name][:idx]
    if "numeric_value" in subject:
        subject["numeric_value"] = subject["numeric_value"][:idx]

    if predict_token_id is not None:
        subject = append_predict_token(subject, censor_date_abspos, predict_token_id)

    return subject


def append_predict_token(
    subject: Dict, censor_date_abspos: float, predict_token_id: int
) -> Dict:
    subject["code"] = torch.cat(
        (
            subject["code"],
            torch.tensor([predict_token_id], dtype=subject["code"].dtype),
        )
    )
    subject["abspos"] = torch.cat(
        (
            subject["abspos"],
            torch.tensor([censor_date_abspos], dtype=subject["abspos"].dtype),
        )
    )
    subject["segment"] = torch.cat(
        (
            subject["segment"],
            torch.tensor(
                [subject["segment"][-1] + 1 if len(subject["segment"]) > 0 else 0],
                dtype=subject["segment"].dtype,
            ),
        )
    )
    if "numeric_value" in subject:
        subject["numeric_value"] = torch.cat(
            (
                subject["numeric_value"],
                torch.tensor([float("nan")], dtype=subject["numeric_value"].dtype),
            )
        )

    age_in_years = float((censor_date_abspos - subject["abspos"][0]) / (365.25 * 24))
    subject["age"] = torch.cat(
        (
            subject["age"],
            torch.tensor(
                [age_in_years],
                dtype=subject["age"].dtype,
            ),
        )
    )
    return subject
