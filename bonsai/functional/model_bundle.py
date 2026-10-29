"""Export model weights and frozen token transformations; verify ZIP extraction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REFERENCE_RESOURCES = {
    "shak_to_region.parquet",
    "c_spec_to_specialty.parquet",
    "sor2_contact_mapping.parquet",
}
NUMERIC_COLUMNS = (
    "code",
    "numeric/bin_1",
    "numeric/bin_99",
    "numeric/bin_edges",
    "numeric/bin_representatives",
    "numeric/hard_minimum",
    "numeric/hard_maximum",
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
SOURCE_FILES = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "template_env",
    "uv.lock",
    "requirements.txt",
    "docs/model_transfer.md",
)
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".ruff_cache"}


def _files(root: Path):
    """Walk without following symbolic links or including environment files."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith(".env") or name.endswith((".pyc", ".pyo")):
                continue
            if path.is_symlink():
                raise ValueError(f"Symbolic links are not supported: {path}")
            yield path
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError(
                    f"Symbolic links are not supported: {Path(directory) / name}"
                )


def _file(path: Path) -> Path:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path


def discover_artifact(folder: Path, filename: str) -> Path:
    """Find exactly one named artifact, refusing missing or ambiguous matches."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Artifact folder does not exist: {folder}")
    matches = [path for path in _files(folder) if path.name == filename]
    if len(matches) != 1:
        detail = ", ".join(str(path.relative_to(folder)) for path in matches)
        raise ValueError(
            f"Expected exactly one {filename} under {folder}; found {len(matches)}"
            f" ({detail}). Select a narrower folder or supply the file explicitly."
        )
    return matches[0]


def _git(root: Path) -> dict:
    def run(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _source_files(bonsai_dir: Path, ehr2meds_dir: Path) -> tuple[dict, dict]:
    """Snapshot package code, configs, resources and installation metadata."""
    entries: dict[str, Path] = {}
    provenance = {}
    for label, root, package in (
        ("BONSAI", Path(bonsai_dir).resolve(), "bonsai"),
        ("ehr2meds", Path(ehr2meds_dir).resolve(), "ehr2meds"),
    ):
        if not (root / "pyproject.toml").is_file() or not (root / package).is_dir():
            raise ValueError(f"{label} must point to the repository root: {root}")
        provenance[label] = _git(root)
        for dirname in (package, "configs", "resources"):
            directory = root / dirname
            if directory.is_symlink():
                raise ValueError(f"Symbolic links are not supported: {directory}")
            if directory.is_dir():
                for path in _files(directory):
                    # Never copy data files simply because they are inside a checkout.
                    allowed = (
                        path.name in REFERENCE_RESOURCES
                        if dirname == "resources"
                        else path.suffix
                        in (
                            {".yaml", ".yml"}
                            if dirname == "configs"
                            else {".py", ".sh"}
                        )
                    )
                    if allowed:
                        entries[
                            f"source/{label}/{path.relative_to(root).as_posix()}"
                        ] = path
        for filename in SOURCE_FILES:
            if (root / filename).exists():
                entries[f"source/{label}/{filename}"] = _file(root / filename)

    return entries, provenance


def _copy_and_hash(src, dst) -> dict:
    """Copy large checkpoint files without holding them in memory."""
    digest = hashlib.sha256()
    size = 0
    while chunk := src.read(1024 * 1024):
        dst.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _export_vocabulary(path: Path, target: Path) -> None:
    """Accept only token-to-integer mappings, not arbitrary torch payloads."""
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
    torch.save(vocabulary, target)


def _export_model(path: Path, target: Path) -> None:
    """Export weights and model hyperparameters, excluding training/callback state."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"Expected a tensor-only state_dict: {path}")
    hparams = {
        key: value
        for key, value in checkpoint.get("hyper_parameters", {}).items()
        if key in MODEL_PARAMETERS
    }
    if not all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in hparams.values()
    ):
        raise ValueError(f"Expected scalar model hyperparameters: {path}")
    exported = {
        "state_dict": {key: value.detach().clone() for key, value in state.items()},
        "hyper_parameters": hparams,
    }
    if "pytorch-lightning_version" in checkpoint:
        exported["pytorch-lightning_version"] = checkpoint["pytorch-lightning_version"]
    torch.save(exported, target)


def _export_transform(path: Path, target: Path, role: str) -> None:
    """Retain only per-code transformation parameters, never raw values or counts."""
    import polars as pl

    frame = pl.read_json(path) if path.suffix == ".json" else pl.read_parquet(path)
    columns = (
        list(NUMERIC_COLUMNS)
        if role == "numeric_metadata"
        else ["code", "adaptive/mapped_code"]
    )
    required = columns[:5] if role == "numeric_metadata" else columns
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


def pack_bundle(
    output: Path,
    *,
    bonsai_dir: Path,
    ehr2meds_dir: Path,
    model_dir: Path,
    data_config: Path,
    numeric_metadata: Path | None,
    code_mapping: Path | None,
    vocabulary: Path | None = None,
    configs: tuple[Path, ...] = (),
) -> Path:
    """Package source, checkpoints, vocabulary and explicitly selected fitted files.

    None for a fitted artifact explicitly means that transform was not used.
    Additional configs must be YAML definitions without patient records.
    Checkpoints export tensor state and model hyperparameters only; fitted
    metadata exports only the standard per-code transformation columns.
    """
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    entries, provenance = _source_files(bonsai_dir, ehr2meds_dir)
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise ValueError(f"Model folder does not exist: {model_dir}")
    model_files = list(_files(model_dir))
    checkpoints = [p for p in model_files if p.suffix == ".ckpt"]
    if not checkpoints:
        raise ValueError(f"No .ckpt checkpoints found in {model_dir}")
    if vocabulary is None:
        candidates = [p for p in model_files if p.name == "vocabulary.pt"]
        if len(candidates) != 1:
            raise ValueError(
                "Supply --vocabulary: model folder must contain exactly one vocabulary.pt"
            )
        vocabulary = candidates[0]
    entries["model/vocabulary.pt"] = _file(vocabulary)
    for path in model_files:
        if path.suffix == ".ckpt" or path.name == "hparams.yaml":
            entries[f"model/run/{path.relative_to(model_dir).as_posix()}"] = path
    # CSVLogger puts checkpoints below the Hydra run directory. Its config
    # records constructor arguments that Lightning hyperparameters omit.
    for parent in (model_dir, *model_dir.parents):
        hydra_dir = parent / "hydra"
        if (hydra_dir / "config.yaml").is_file():
            for path in _files(hydra_dir):
                if path.name in {"config.yaml", "overrides.yaml"}:
                    entries[
                        f"settings/model_hydra/{path.relative_to(hydra_dir).as_posix()}"
                    ] = path
            break
    entries["settings/bonsai_data" + Path(data_config).suffix] = _file(data_config)
    artifacts = {}
    for role, path in (
        ("numeric_metadata", numeric_metadata),
        ("code_mapping", code_mapping),
    ):
        if path is None:
            artifacts[role] = None
        else:
            path = _file(path)
            if path.suffix not in {".json", ".parquet"}:
                raise ValueError(f"{role} must be JSON or Parquet")
            target = f"preprocessing/{role}{path.suffix}"
            entries[target] = path
            artifacts[role] = target
    for index, path in enumerate(configs):
        path = _file(path)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError(
                "Additional configs must be YAML; patient data attachments are unsupported"
            )
        entries[f"settings/preprocessing/{index:03d}/{path.name}"] = path

    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination and publish only a fully written archive.
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        # These are new export files; source artifacts are never modified.
        for index, (name, path) in enumerate(list(entries.items())):
            if (
                path.suffix == ".ckpt"
                or name.startswith("preprocessing/")
                or name == "model/vocabulary.pt"
            ):
                exported = temporary / f"{index}{path.suffix}"
                if name == "model/vocabulary.pt":
                    _export_vocabulary(path, exported)
                elif path.suffix == ".ckpt":
                    _export_model(path, exported)
                else:
                    _export_transform(path, exported, name.split("/")[1].split(".")[0])
                entries[name] = exported
        archive = temporary / "bundle.zip"
        inventory = {}
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as bundle:
            for name, path in sorted(entries.items()):
                with (
                    path.open("rb") as src,
                    bundle.open(name, "w", force_zip64=True) as dst,
                ):
                    inventory[name] = _copy_and_hash(src, dst)
            manifest = {
                "format_version": 1,
                "repositories": provenance,
                "scope": "Model and frozen token transformations; no source-site patient records or cohorts.",
                "artifacts": artifacts,
                "files": inventory,
                "notes": "See source/BONSAI/docs/model_transfer.md for restoration steps and limitations.",
            }
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        os.link(
            archive, output
        )  # Refuse to overwrite even if another process created it.
    return output


def unpack_bundle(archive: Path, destination: Path) -> Path:
    """Verify and extract into a new directory; reject traversal and symlinks.

    Failed verification leaves no destination. Existing directories are never merged.
    """
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name).parts
            if (
                not parts
                or PurePosixPath(name).is_absolute()
                or any(p in {".", ".."} or ":" in p for p in parts)
                or "\\" in name
                or name != PurePosixPath(name).as_posix()
                or info.is_dir()
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {name}")
        manifest = json.loads(bundle.read("manifest.json"))
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported bundle format")
        inventory = manifest["files"]
        if set(names) != set(inventory) | {"manifest.json"}:
            raise ValueError("Archive contents do not match manifest")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "contents"
            staging.mkdir()
            for info in infos:
                target = staging / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as src, target.open("xb") as dst:
                    actual = _copy_and_hash(src, dst)
                if (
                    info.filename != "manifest.json"
                    and inventory[info.filename] != actual
                ):
                    raise ValueError(f"Checksum mismatch: {info.filename}")
            # Reserve destination to avoid replacing an existing directory.
            destination.mkdir()
            try:
                for child in staging.iterdir():
                    shutil.move(str(child), destination / child.name)
            except BaseException:
                shutil.rmtree(destination)
                raise
    return destination
