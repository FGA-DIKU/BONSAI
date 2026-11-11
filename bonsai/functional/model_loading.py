"""Load an extracted bundle for inference; preprocessing remains site-local."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from bonsai.functional.model_bundle import load_vocabulary, validate_manifest
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiFinetune,
    BonsaiPretrain,
    BonsaiValueFinetune,
    BonsaiValuePretrain,
)

MODEL_CLASSES = {
    model.__name__: model
    for model in (
        BonsaiPretrain,
        BonsaiValuePretrain,
        BonsaiFinetune,
        BonsaiValueFinetune,
    )
}


@dataclass
class LoadedBundle:
    model: BonsaiBase
    vocabulary: dict[str, int]
    numeric_metadata: Path | None
    code_mapping: Path | None


def load_bundle(
    directory: Path, device: str = "cpu", attn_type: str | None = None
) -> LoadedBundle:
    """Load the bundled network strictly, in evaluation mode, on the chosen device.

    Use unpack_bundle first for ZIP files. An explicit attn_type="sdpa" override
    permits loading without FlashAttention. The vocabulary and transformations
    are fixed inputs to preprocessing the destination site's own records.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Unpack the bundle first, then supply its extracted directory")
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    for name in manifest["files"]:
        path = directory / name
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Missing or linked bundle file: {name}")
    name = manifest["model"]["class"]
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model class: {name}")
    parameters = dict(manifest["model"]["parameters"])
    if attn_type is not None:
        parameters["attn_type"] = attn_type
    vocabulary = load_vocabulary(
        directory / "model/vocabulary.pt", parameters["vocab_size"]
    )
    checkpoint = torch.load(
        directory / "model/model.ckpt", map_location="cpu", weights_only=True
    )
    model = MODEL_CLASSES[name](**parameters)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    artifacts = {
        role: directory / path if path is not None else None
        for role, path in manifest["artifacts"].items()
    }
    return LoadedBundle(
        model.to(device).eval(),
        vocabulary,
        artifacts["numeric_metadata"],
        artifacts["code_mapping"],
    )
