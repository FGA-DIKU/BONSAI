"""Create or extract a portable BONSAI/ehr2meds model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from bonsai.functional.model_bundle import discover_artifact, pack_bundle, unpack_bundle


def _fitted_file(
    explicit: Path | None, unused: bool, folder: Path | None, filename: str
):
    """Prefer an explicit file; otherwise discover the original fitted artifact."""
    if unused:
        return None
    if explicit is not None:
        return explicit
    if folder is None:
        raise ValueError(
            f"Supply --ehr2meds-output-dir or an explicit file for {filename}"
        )
    return discover_artifact(folder, filename)


def _pack(args) -> Path:
    """Resolve folder defaults before calling the file-based packaging function."""
    vocabulary = args.vocabulary
    if vocabulary is None and args.data_dir is not None:
        vocabulary = discover_artifact(args.data_dir, "vocabulary.pt")
    data_config = args.data_config
    if data_config is None:
        if args.data_dir is None:
            raise ValueError("Supply --data-dir or --data-config")
        data_config = args.data_dir / "logs/hydra/config.yaml"
        if not data_config.is_file():
            raise ValueError(
                "No logs/hydra/config.yaml in --data-dir; supply --data-config"
            )
    return pack_bundle(
        args.output,
        bonsai_dir=args.bonsai_dir,
        ehr2meds_dir=args.ehr2meds_dir,
        model_dir=args.model_dir,
        vocabulary=vocabulary,
        data_config=data_config,
        numeric_metadata=_fitted_file(
            args.numeric_metadata,
            args.no_numeric_metadata,
            args.ehr2meds_output_dir,
            "numeric_metadata.json",
        ),
        code_mapping=_fitted_file(
            args.code_mapping,
            args.no_code_mapping,
            args.ehr2meds_output_dir,
            "adaptive_code_mapping.parquet",
        ),
        configs=tuple(args.config),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Package model and preprocessing artifacts")
    pack.add_argument(
        "--ehr2meds-dir", type=Path, required=True, help="Source checkout"
    )
    pack.add_argument(
        "--bonsai-dir", type=Path, default=Path(__file__).resolve().parents[2]
    )
    pack.add_argument("--ehr2meds-output-dir", type=Path, help="Fitted metadata folder")
    pack.add_argument(
        "--model-dir", type=Path, required=True, help="Selected model run"
    )
    pack.add_argument("--data-dir", type=Path, help="BONSAI processed dataset folder")
    overrides = pack.add_argument_group("File overrides for custom layouts")
    overrides.add_argument("--vocabulary", type=Path, help="Original vocabulary.pt")
    overrides.add_argument(
        "--data-config", type=Path, help="Original create-data config"
    )
    for role in ("numeric-metadata", "code-mapping"):
        group = overrides.add_mutually_exclusive_group()
        group.add_argument(f"--{role}", type=Path, help="Original fitted JSON/Parquet")
        group.add_argument(
            f"--no-{role}", action="store_true", help="Transform was unused"
        )
    pack.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Preprocessing YAML definition without patient records; repeatable",
    )
    pack.add_argument("--output", type=Path, required=True)
    unpack = commands.add_parser("unpack", help="Verify and extract into a new folder")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "unpack":
            result = unpack_bundle(args.archive, args.destination)
        else:
            result = _pack(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(result)


if __name__ == "__main__":
    main()
