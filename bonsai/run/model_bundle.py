"""Pack explicit model/token files, unpack a bundle, or check that its model loads."""

import argparse
from pathlib import Path

from bonsai.functional.model_bundle import pack_bundle, unpack_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Package explicitly selected files")
    pack.add_argument("--checkpoint", type=Path, required=True)
    pack.add_argument("--vocabulary", type=Path, required=True)
    pack.add_argument("--data-config", type=Path, required=True)
    for role in ("numeric-metadata", "code-mapping"):
        group = pack.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{role}", type=Path)
        group.add_argument(
            f"--no-{role}", action="store_true", help="Transform was unused"
        )
    pack.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Additional preprocessing YAML; repeatable",
    )
    pack.add_argument("--output", type=Path, required=True)
    unpack = commands.add_parser("unpack", help="Extract into a new directory")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("destination", type=Path)
    load = commands.add_parser("load", help="Load an extracted bundle for inference")
    load.add_argument("directory", type=Path)
    load.add_argument("--device", default="cpu")
    load.add_argument("--attn-type", choices=("sdpa", "flash"))
    args = parser.parse_args()
    try:
        if args.command == "pack":
            print(
                pack_bundle(
                    args.output,
                    checkpoint=args.checkpoint,
                    vocabulary=args.vocabulary,
                    data_config=args.data_config,
                    numeric_metadata=args.numeric_metadata,
                    code_mapping=args.code_mapping,
                    configs=tuple(args.config),
                )
            )
        elif args.command == "unpack":
            print(unpack_bundle(args.archive, args.destination))
        else:
            from bonsai.functional.model_loading import load_bundle

            bundle = load_bundle(
                args.directory, device=args.device, attn_type=args.attn_type
            )
            print(
                f"Loaded {type(bundle.model).__name__}; vocabulary={len(bundle.vocabulary)} tokens"
            )
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
