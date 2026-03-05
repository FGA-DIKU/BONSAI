import torch
from typing import List, Dict

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
    output["subject_id"] = torch.tensor(collected["subject_id"])

    return output

