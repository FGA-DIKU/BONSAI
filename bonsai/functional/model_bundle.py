"""Bundle explicit model/token files; no repository discovery or source snapshots."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_NUMERIC_COLUMNS = (
    "code",
    "numeric/bin_1",
    "numeric/bin_99",
    "numeric/bin_edges",
    "numeric/bin_representatives",
)


MODEL_PARAMETERS = {
    "vocab_size",
    "max_seqlen",
    "hidden_size",
    "num_layers",
    "bias",
    "dropout",
    "attention_dropout",
    "causal",
    "attn_type",
    "value_embedding_mode",
    "predict_token_id",
}


def safe_member_name(name: str) -> bool:
    """Bundle members are canonical relative POSIX paths."""
    path = PurePosixPath(name)
    return bool(path.parts) and not (
        path.is_absolute()
        or any(part in {".", ".."} or ":" in part for part in path.parts)
        or "\\" in name
        or name != path.as_posix()
    )


def load_vocabulary(path: Path, vocab_size: int | None = None) -> dict:
    """Read unique token IDs; require one token per embedding row when size is given."""
    import torch

    vocabulary = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(vocabulary, dict)
        or not vocabulary
        or not all(
            isinstance(token, str) and type(index) is int and index >= 0
            for token, index in vocabulary.items()
        )
        or len(set(vocabulary.values())) != len(vocabulary)
    ):
        raise ValueError(
            "Expected a vocabulary containing unique token-to-integer mappings"
        )
    if vocab_size is not None and (
        len(vocabulary) != vocab_size or max(vocabulary.values()) >= vocab_size
    ):
        raise ValueError("Vocabulary must cover the model's embedding table exactly.")
    return vocabulary


def model_payload(state_dict: dict, hparams: dict) -> dict:
    """Keep independent CPU weight tensors and model constructor scalars only."""
    import torch

    state = {
        key: value for key, value in state_dict.items() if key.startswith("model.")
    }
    if not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("Expected a tensor-only model state_dict")
    parameters = {
        key: value for key, value in hparams.items() if key in MODEL_PARAMETERS
    }
    if not all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in parameters.values()
    ):
        raise ValueError("Expected scalar model hyperparameters")
    return {
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in state.items()
        },
        "hyper_parameters": parameters,
    }


def export_transform(path: Path, target: Path, role: str) -> None:
    """Retain only per-code transformation parameters, never raw values or counts."""
    import polars as pl

    frame = pl.read_json(path) if path.suffix == ".json" else pl.read_parquet(path)
    if role == "numeric_metadata":
        required = REQUIRED_NUMERIC_COLUMNS
        columns = list(required) + ["numeric/hard_minimum", "numeric/hard_maximum"]
    elif role == "code_mapping":
        required = columns = ["code", "adaptive/mapped_code"]
    else:
        raise ValueError(f"Unknown transformation role: {role}")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{role} missing standard transformation columns: {sorted(missing)}"
        )
    frame = frame.select([name for name in columns if name in frame.columns])
    if frame["code"].n_unique() != frame.height:
        raise ValueError(f"{role} must contain one transformation per code")
    if role == "code_mapping" and frame.schema["adaptive/mapped_code"] != pl.String:
        raise ValueError("Adaptive mapped codes must be strings")
    if frame.schema["code"] != pl.String:
        raise ValueError("Transformation codes must be strings")
    if role == "numeric_metadata":
        for name, dtype in frame.schema.items():
            if name != "code" and not (
                dtype.is_numeric()
                or (isinstance(dtype, pl.List) and dtype.inner.is_numeric())
                or dtype == pl.Null
            ):
                raise ValueError(f"Expected numeric transformation parameters: {name}")
    frame.write_json(target) if target.suffix == ".json" else frame.write_parquet(
        target
    )


def validate_manifest(manifest: dict) -> None:
    """Require the small bundle layout and safe relative asset paths."""
    if manifest.get("format_version") != 2:
        raise ValueError(
            "Unsupported bundle format; repack using the explicit-file command."
        )
    files = manifest["files"]
    if not all(safe_member_name(name) for name in files):
        raise ValueError("Unsafe bundle path")
    if not {"model/model.ckpt", "model/vocabulary.pt"}.issubset(files):
        raise ValueError("Bundle must contain a checkpoint and vocabulary")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != {"numeric_metadata", "code_mapping"}:
        raise ValueError(
            "Declare numeric_metadata and code_mapping; use null for unused transforms"
        )
    for path in artifacts.values():
        if path is not None and path not in files:
            raise ValueError(f"Unlisted artifact: {path}")


def export_checkpoint(checkpoint: Path, target: Path) -> dict:
    """Save cleaned weights; return the model class and constructor settings."""
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload = model_payload(saved["state_dict"], saved["hyper_parameters"])
    parameters = payload["hyper_parameters"].copy()
    parameters.setdefault("attention_dropout", 0.0)  # Unused during inference.
    value_model = bool(parameters.get("value_embedding_mode"))
    if not value_model:
        parameters.pop("value_embedding_mode", None)
    if "model.finetune_head.weight" in payload["state_dict"]:
        model_class = "BonsaiValueFinetune" if value_model else "BonsaiFinetune"
    elif "model.pretrain_head.weight" in payload["state_dict"]:
        model_class = "BonsaiValuePretrain" if value_model else "BonsaiPretrain"
    else:
        raise ValueError("Checkpoint has no supported pretraining or fine-tuning head")
    torch.save(payload, target)
    return {"class": model_class, "parameters": parameters}


def pack_bundle(
    output: Path,
    checkpoint: Path,
    vocabulary: Path,
    data_config: Path,
    numeric_metadata: Path | None,
    code_mapping: Path | None,
    configs: tuple[Path, ...] = (),
) -> Path:
    """ZIP one checkpoint, vocabulary, fitted transforms and selected YAML configs.

    None explicitly declares an unused transform. Checkpoint training state and
    raw numeric values/counts are discarded. Original files stay unchanged.
    """
    import torch

    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "model").mkdir()
        model_spec = export_checkpoint(checkpoint, stage / "model/model.ckpt")
        vocab = load_vocabulary(vocabulary, model_spec["parameters"]["vocab_size"])
        torch.save(vocab, stage / "model/vocabulary.pt")
        files = ["model/model.ckpt", "model/vocabulary.pt"]
        for index, config in enumerate((data_config,) + configs):
            config = Path(config)
            if config.suffix not in {".yaml", ".yml"}:
                raise ValueError(
                    "Configs must be YAML definitions without patient records"
                )
            target = f"settings/{index:03d}_{config.name}"
            (stage / target).parent.mkdir(exist_ok=True)
            shutil.copyfile(config, stage / target)
            files.append(target)
        artifacts = {}
        for role, path in (
            ("numeric_metadata", numeric_metadata),
            ("code_mapping", code_mapping),
        ):
            artifacts[role] = None
            if path is not None:
                path = Path(path)
                if path.suffix not in {".json", ".parquet"}:
                    raise ValueError(f"{role} must be JSON or Parquet")
                target = f"preprocessing/{role}{path.suffix}"
                (stage / target).parent.mkdir(exist_ok=True)
                export_transform(path, stage / target, role)
                artifacts[role] = target
                files.append(target)
        manifest = {
            "format_version": 2,
            "model": model_spec,
            "artifacts": artifacts,
            "files": files,
        }
        validate_manifest(manifest)
        archive = stage / "bundle.zip"
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as bundle:
            for name in files:
                bundle.write(stage / name, name)
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        os.link(archive, output)  # Publish complete archives; refuse overwrites.
    return output


def unpack_bundle(archive: Path, destination: Path) -> Path:
    """Extract a ZIP into a new folder; reject unsafe paths and unlisted files."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for member in members:
            if (
                not safe_member_name(member.filename)
                or member.is_dir()
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        manifest = json.loads(bundle.read("manifest.json"))
        validate_manifest(manifest)
        if set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Archive contents do not match manifest")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            stage = Path(temporary)
            for member in members:
                target = stage / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)  # ZIP CRC checks run while reading.
            destination.mkdir()
            try:
                for child in stage.iterdir():
                    shutil.move(str(child), destination / child.name)
            except BaseException:
                shutil.rmtree(destination)
                raise
    return destination
