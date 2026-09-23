from typing import Dict, List

import torch


def dynamic_padding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collected = {key: [] for key in batch[0]}
    for sample in batch:
        for key, val in sample.items():
            collected[key].append(val)

    output = {}
    for embed_name in ["code", "abspos", "age", "segment", "attention_mask"]:
        output[embed_name] = torch.nn.utils.rnn.pad_sequence(
            collected[embed_name], batch_first=True
        )
    output["target"] = torch.nn.utils.rnn.pad_sequence(
        collected["target"], batch_first=True, padding_value=-100
    )
    if "numeric_value" in collected:
        output["numeric_value"] = torch.nn.utils.rnn.pad_sequence(
            collected["numeric_value"], batch_first=True, padding_value=float("nan")
        )
    if "numeric_target" in collected:
        output["numeric_target"] = torch.nn.utils.rnn.pad_sequence(
            collected["numeric_target"], batch_first=True, padding_value=float("nan")
        )
    output["subject_id"] = torch.tensor(collected["subject_id"])

    return output


def tte_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    output = dynamic_padding(batch)
    output["duration"] = torch.nn.utils.rnn.pad_sequence(
        [sample["duration"] for sample in batch], batch_first=True
    )
    return output
