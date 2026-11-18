#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Set headless backend before importing pyplot.
import matplotlib.pyplot as plt

LOG = logging.getLogger("bonsai_umap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BONSAI held-out UMAP scatter plots: predicted risk and observed outcome (no smoothing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained FINETUNED best.ckpt (not pretrained checkpoint)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Processed dataset root OR tokenized held_out parquet directory")
    parser.add_argument("--held-out-parquets", type=Path, default=None,
                        help="Optional held-out parquet directory if separate from --dataset")
    parser.add_argument("--vocabulary", type=Path, default=None,
                        help="Optional original training vocabulary.pt for compatibility validation")
    parser.add_argument("--population", type=Path, default=None,
                        help="Optional population_full.csv for an additional cohort filter")
    parser.add_argument("--outcomes", type=Path, default=None,
                        help="Parquet file containing BONSAI outcome dates and split")
    parser.add_argument("--finetune-config", type=Path, default=None,
                        help="Original Hydra config.yaml (or auto-discovered next to checkpoint)")
    parser.add_argument("--output-dir", type=Path, default=Path("umap_results"))
    parser.add_argument("--max-len", type=int, default=None,
                        help="Override training.max_len if original Hydra config is missing")
    parser.add_argument("--label-start-hours", type=int, default=None,
                        help="Override labels.start_include (hours)")
    parser.add_argument("--label-end-days", type=int, default=None,
                        help="Override labels.end_include (days)")
    parser.add_argument("--horizon-days", type=float, default=None,
                        help="Optional check against the training label window; does not alter labels")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--attn-type", choices=["flash", "sdpa"], default=None,
                        help="Optional checkpoint attention backend override; validate parity")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto",
                        help="CUDA autocast for FlashAttention: auto chooses BF16 if supported, else FP16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--risk-max", type=float, default=1.0,
                        help="Maximum of predicted-risk color scale")
    parser.add_argument("--save-patient-table", action="store_true",
                        help="Save identifiable patient-level CSV (sensitive data)")
    return parser


def get_train_config(checkpoint: Path, override: Path | None):
    """Recover original max_len/label window if Hydra stored config next to ckpt."""
    from omegaconf import OmegaConf
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    if override:
        candidates = [override.expanduser().resolve()]
    else:
        # Hydra config path is 'hydra/config.yaml' for recent BONSAI, '.hydra' for older runs.
        candidates = [p / hidden / "config.yaml"
                      for p in (checkpoint.parent, checkpoint.parent.parent)
                      for hidden in ("hydra", ".hydra")]
    for path in candidates:
        if path.is_file():
            LOG.info("Using original finetune config: %s", path)
            return OmegaConf.load(path), path
    if override:
        raise FileNotFoundError(f"Finetuning config not found: {override}")
    LOG.warning("No original Hydra config found next to checkpoint. Training max_len and labels "
                "CANNOT be recovered from the model weights alone. Use --finetune-config "
                "(recommended), or explicitly set --max-len and label-window options.")
    return None, None


def config_setting(cfg, path, default=None):
    if cfg is None:
        return default
    from omegaconf import OmegaConf
    value = OmegaConf.select(cfg, path, default=default, throw_on_missing=False)
    return value


def resolve_inputs(args, train_cfg):
    """Resolve tokenized patient shards, single target outcomes and optional artifacts.

    Explicit --dataset + --outcomes + --checkpoint suffice when the original
    vocabulary and population files are unavailable. In that case the model's
    stored vocab_size and predict_token_id define the token IDs.
    """
    supplied = args.dataset.expanduser().resolve()
    if not supplied.exists():
        raise FileNotFoundError(f"Dataset path missing: {supplied}")
    if supplied.is_file() and supplied.name == "subject_data_held_out.pt":
        root = supplied.parent
    elif supplied.is_dir() and supplied.name == "held_out":
        root = supplied.parent
    else:
        root = supplied
    if not root.is_dir():
        raise ValueError("--dataset must be a processed dataset root, a held_out parquet directory, or subject_data_held_out.pt")

    if args.held_out_parquets is not None:
        patient_path = args.held_out_parquets.expanduser().resolve()
    elif supplied.is_file():
        patient_path = supplied
    elif supplied.is_dir() and any(supplied.glob("*.parquet")):
        patient_path = supplied
    elif (root / "subject_data_held_out.pt").is_file():
        patient_path = root / "subject_data_held_out.pt"
    elif (root / "held_out").is_dir():
        patient_path = root / "held_out"
    else:
        raise FileNotFoundError(
            f"No held_out tokenized parquet files or subject_data_held_out.pt found under {supplied}. "
            "Supply --dataset /path/to/held_out or --held-out-parquets explicitly."
        )
    if patient_path.is_dir():
        files = sorted(patient_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet shards found directly under {patient_path}")
        import polars as pl
        for shard in files:
            columns = set(pl.read_parquet_schema(shard))
            expected = {"subject_id", "code", "age", "abspos", "segment"}
            if expected - columns:
                # One separate outcomes file in the same directory is allowed:
                # it will be excluded in make_predict_loader before BONSAI loads.
                outcome_cols = {"subject_id", "outcome_date", "index_date", "censor_date", "split"}
                if outcome_cols <= columns and args.outcomes is not None and shard == args.outcomes.expanduser().resolve():
                    continue
                raise ValueError(
                    f"{shard} does not contain tokenized BONSAI columns {sorted(expected)}. "
                    "Pass the held-out TOKENIZED parquet shard directory as --dataset and "
                    "the single outcome table as --outcomes. Raw MEDS data needs tokenization first."
                )
    elif not patient_path.is_file() or patient_path.suffix != ".pt":
        raise ValueError("Patient input must be subject_data_held_out.pt or a directory of tokenized .parquet files")

    vocab_path = args.vocabulary.expanduser().resolve() if args.vocabulary else root / "vocabulary.pt"
    if not vocab_path.is_file():
        if args.vocabulary:
            raise FileNotFoundError(f"Explicit --vocabulary not found: {vocab_path}")
        LOG.warning("No vocabulary.pt found; using checkpoint's stored vocab_size and predict_token_id. "
                    "Please ensure the parquet codes use the SAME vocabulary as training.")
        vocab_path = None

    population_path = args.population.expanduser().resolve() if args.population else root / "cohorts" / "population_full.csv"
    if not population_path.is_file():
        if args.population:
            raise FileNotFoundError(f"Explicit --population not found: {population_path}")
        LOG.info("No population_full.csv; use subject IDs from the held-out outcomes as the cohort")
        population_path = None

    if args.outcomes is not None:
        outcome_path = args.outcomes.expanduser().resolve()
    else:
        outcome_name = config_setting(train_cfg, "outcome")
        outcome_root = root / "outcomes"
        outcome_path = None
        if outcome_name:
            stem = str(outcome_name)
            for cand in (outcome_root / (stem + ".parquet"), outcome_root / stem):
                if cand.is_file() or cand.is_dir():
                    outcome_path = cand
                    break
        if outcome_path is None:
            single_files = sorted(outcome_root.glob("*.parquet")) if outcome_root.is_dir() else []
            single_dirs = (sorted(p for p in outcome_root.iterdir() if p.is_dir() and list(p.rglob("*.parquet")))
                           if outcome_root.is_dir() else [])
            candidates = single_files + single_dirs
            if len(candidates) != 1:
                raise ValueError(
                    f"Found {len(candidates)} possible outcomes in {outcome_root}. "
                    "Specify --outcomes /path/to/ONE_target.parquet."
                )
            outcome_path = candidates[0]
    if not outcome_path.exists():
        raise FileNotFoundError(f"Outcome file/folder missing: {outcome_path}")
    if outcome_path.is_dir() and not list(outcome_path.rglob("*.parquet")):
        raise FileNotFoundError(f"Outcome folder contains no .parquet files: {outcome_path}")
    return root, patient_path, vocab_path, population_path, outcome_path


def read_outcome_shards(outcome_path: Path):
    """Read a single outcome table, optionally sharded over parquet files."""
    import polars as pl
    if outcome_path.is_file():
        return pl.read_parquet(outcome_path)
    files = sorted(outcome_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No outcome parquet files in {outcome_path}")
    LOG.info("Reading %d outcome shards for ONE prediction target", len(files))
    return pl.scan_parquet([str(f) for f in files]).collect()


def get_model_and_module(checkpoint: Path, vocabulary: dict | None, attention_override: str | None):
    """Construct BONSAI from finetuned checkpoint hyperparameters, strictly load weights."""
    import torch
    from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
    from bonsai.modules.networks.bonsai_nets import BonsaiFinetune, BonsaiValueFinetune

    # Only trust your own checkpoint: this contains deserialized Python objects.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = dict(checkpoint_data.get("hyper_parameters", {}))
    state = checkpoint_data.get("state_dict", {})
    if not hp or not any(k.startswith("model.finetune_head.") for k in state):
        raise ValueError("Not a BONSAI FINETUNED checkpoint (missing model.finetune_head weights)")
    for key in ("predict_token_id", "vocab_size", "max_seqlen", "hidden_size", "num_layers", "bias", "dropout", "causal", "attn_type"):
        if key not in hp:
            raise ValueError(f"Finetuned checkpoint is missing hyperparameter: {key}")
    if vocabulary is not None:
        if "[CLS]" not in vocabulary:
            raise ValueError("vocabulary.pt does not contain a [CLS] token")
        if int(vocabulary["[CLS]"]) != int(hp["predict_token_id"]):
            raise ValueError("Vocabulary's [CLS] ID differs from checkpoint; wrong processed dataset?")
        if len(vocabulary) != int(hp["vocab_size"]):
            raise ValueError("Vocabulary size differs from checkpoint; wrong processed dataset?")

    is_value = hp.get("value_embedding_mode") is not None
    model_type = BonsaiValueFinetune if is_value else BonsaiFinetune
    attn_type = attention_override or hp["attn_type"]
    if attention_override and attention_override != hp["attn_type"]:
        LOG.warning("Changing attention backend %s -> %s; validate numerical parity", hp["attn_type"], attn_type)
    model_args = {
        "vocab_size": int(hp["vocab_size"]),
        "max_seqlen": int(hp["max_seqlen"]),
        "hidden_size": int(hp["hidden_size"]),
        "num_layers": int(hp["num_layers"]),
        "bias": bool(hp["bias"]),
        "dropout": float(hp["dropout"]),
        # attention_dropout is NOT stored in checkpoint; eval mode disables it.
        "attention_dropout": 0.0,
        "causal": bool(hp["causal"]),
        "attn_type": attn_type,
        "predict_token_id": int(hp["predict_token_id"]),
    }
    if is_value:
        model_args["value_embedding_mode"] = hp["value_embedding_mode"]
    model = model_type(**model_args)
    saved_pos_weight = hp.get("pos_weight")
    pos_weight = None if saved_pos_weight is None else torch.tensor(float(saved_pos_weight))
    module = FinetuneModule.load_from_checkpoint(
        str(checkpoint), model=model, pos_weight=pos_weight,
        predictions_output_path=None, strict=True, map_location="cpu"
    )
    module.eval()
    return module, hp


def make_predict_loader(patient_path: Path, population_path: Path | None, outcome_path: Path,
                        predict_token_id: int, max_len: int,
                        start_include: dict, end_include: dict | None,
                        batch_size: int, num_workers: int):
    """Re-use BONSAI's outcome binarization, patient preprocessing, and collator.

    The outcome file is a SINGLE target. All model inputs are selected by
    patient ID, not row position. The parquet shards are already tokenized.
    """
    import polars as pl
    from bonsai.functional.outcomes import binarize_outcomes, finalize_outcomes

    df = read_outcome_shards(outcome_path)
    required = {"split", "subject_id", "index_date", "outcome_date", "censor_date"}
    if required - set(df.columns):
        raise ValueError(f"Outcome file missing BONSAI columns: {sorted(required - set(df.columns))}")
    held_out = df.filter(pl.col("split") == "held_out")
    if held_out.is_empty():
        raise ValueError("Outcome file contains no patients with split='held_out'")
    if held_out["subject_id"].n_unique() != held_out.height:
        raise ValueError("Outcome file has duplicate held-out subject_id rows; expected one row per patient")
    filtered = binarize_outcomes(held_out, start_include=start_include, end_include=end_include)
    outcome_lookup = finalize_outcomes(filtered)
    if not outcome_lookup:
        raise ValueError("No usable held-out patients after BONSAI label window filtering")

    if patient_path.is_file():
        # The BONSAI Lightning datamodule natively loads precomputed .pt files.
        from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
        if population_path is None:
            # Reuse BONSAI data/label conversion without inventing a population file.
            import torch
            from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
            from bonsai.functional.collate import dynamic_padding
            from torch.utils.data import DataLoader
            subjects = torch.load(patient_path, map_location="cpu", weights_only=False)
            selected = [s for s in subjects if int(s["subject_id"]) in outcome_lookup]
            if not selected:
                raise ValueError("No overlapping subject IDs between .pt patient data and outcome file")
            selected.sort(key=lambda sub: int(sub["subject_id"]))
            bg_len = int((selected[0]["segment"] == 1).sum())
            dataset = FinetuneDataset(selected, outcome_lookup, predict_token_id, bg_len, max_len)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=False, persistent_workers=num_workers > 0,
                                shuffle=False, drop_last=False, collate_fn=dynamic_padding)
        else:
            dm = FinetuneDataModule(
                path_train_data=str(patient_path), path_val_data=str(patient_path),
                path_predict_data=str(patient_path), path_population=str(population_path),
                batch_size=batch_size, num_workers=num_workers,
                predict_token_id=predict_token_id, max_len=max_len,
                train_outcomes={}, val_outcomes={}, predict_outcomes=outcome_lookup,
            )
            dm.setup("predict")
            dataset, loader = dm.predict_dataset, dm.predict_dataloader()
    else:
        from torch.utils.data import DataLoader
        from bonsai.functional.subject_data import prepare_subject_data, filter_subject_data
        from bonsai.functional.collate import dynamic_padding
        from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
        import tempfile

        LOG.info("Reading tokenized held-out parquet shards from %s", patient_path)
        # BONSAI prepare_subject_data reads EVERY *.parquet in a directory.
        # Exclude an outcomes.parquet stored alongside patient shards.
        shard_files = sorted(patient_path.glob("*.parquet"))
        shard_files = [p for p in shard_files if p.resolve() != outcome_path.resolve()]
        if not shard_files:
            raise ValueError("No tokenized patient shards remain after excluding the outcome parquet")
        if len(shard_files) < len(list(patient_path.glob("*.parquet"))):
            with tempfile.TemporaryDirectory(prefix="bonsai_umap_shards_") as td:
                staging = Path(td)
                for idx, shard in enumerate(shard_files):
                    (staging / f"{idx:06}.parquet").symlink_to(shard.resolve())
                subjects = prepare_subject_data(split_path=staging)
        else:
            subjects = prepare_subject_data(split_path=patient_path)
        selected = [s for s in subjects if int(s["subject_id"]) in outcome_lookup]
        if not selected:
            raise ValueError("No matching subject IDs between tokenized patient shards and held-out outcomes")
        # BONSAI groups within shards; if a patient crosses shards, the model
        # would see an incomplete history. Fail explicitly instead of guessing.
        ids = [int(s["subject_id"]) for s in selected]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "A patient occurs in several tokenized shards. BONSAI's loader groups within "
                "each shard, so these files must first be repartitioned by subject_id."
            )
        if population_path is not None:
            population_ids = pl.read_csv(population_path)["subject_id"].to_list()
            selected = filter_subject_data(selected, population_ids)
            if not selected:
                raise ValueError("No held-out patients remain after population_full.csv filtering")
        selected.sort(key=lambda sub: int(sub["subject_id"]))
        background_length = int((selected[0]["segment"] == 1).sum())
        dataset = FinetuneDataset(
            selected, outcomes=outcome_lookup, predict_token_id=predict_token_id,
            background_length=background_length, max_len=max_len,
        )
        loader = DataLoader(
            dataset, num_workers=num_workers, batch_size=batch_size,
            pin_memory=False, persistent_workers=num_workers > 0,
            shuffle=False, drop_last=False, collate_fn=dynamic_padding,
        )
    if len(dataset) < 4:
        raise ValueError(f"Need at least 4 held-out patients to plot UMAP, found {len(dataset)}")
    records = {int(row["subject_id"]): row for row in filtered.to_dicts()}
    return loader, records, len(dataset)


def infer_in_one_pass(module, dataloader, device, attention_backend="sdpa", amp_dtype="auto"):
    """Extract BONSAI embeddings and scores in one pass with FlashAttention-safe CUDA autocast.

    FlashAttention needs BF16/FP16 QKV. Loading Lightning checkpoint without a Trainer
    does not automatically enable the mixed precision used during finetuning.
    """
    import torch
    from contextlib import nullcontext

    dev = torch.device(device)
    if attention_backend == "flash" and dev.type != "cuda":
        raise RuntimeError("BONSAI FlashAttention requires a CUDA GPU; use --device cuda "
                           "or --attn-type sdpa to switch backend (validate output parity).")

    use_amp = attention_backend == "flash" or amp_dtype != "auto"
    amp_context = nullcontext
    if use_amp:
        if dev.type != "cuda":
            raise RuntimeError("Mixed-precision inference requires --device cuda in this script")
        if amp_dtype == "auto":
            precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            precision = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        if precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA GPU does not support bfloat16; try --amp-dtype fp16")
        LOG.info("Inference autocast: %s (model parameters are not explicitly downcast)", precision)
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=precision)

    model = module.model
    if not hasattr(model, "finetune_head"):
        raise TypeError("Model lacks finetune_head: cannot extract prediction-token embeddings")
    module.to(device)
    module.eval()
    capture = []

    def record_embedding(head, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 2:
            raise RuntimeError("Expected [batch, hidden_dim] input to finetune_head")
        capture.append(inputs[0].detach().float().cpu())

    hook = model.finetune_head.register_forward_pre_hook(record_embedding)
    ids, embeddings, probs, labels = [], [], [], []
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader):
                moved = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                capture.clear()
                # FlashAttention asserts qkv.dtype is FP16/BF16; CUDA autocast makes
                # the Wqkv linear operation output a supported dtype.
                with amp_context():
                    prediction = module.predict_step(moved, batch_index)
                batch_ids = prediction["subject_id"].detach().cpu().reshape(-1).numpy()
                scores = prediction["prob"].detach().float().cpu().reshape(-1).numpy()
                outcomes = prediction["label"].detach().cpu().reshape(-1).numpy().astype(int)
                if len(capture) != 1 or capture[0].shape[0] != len(batch_ids):
                    raise RuntimeError("Prediction-head embeddings do not match this patient's batch")
                if len(scores) != len(batch_ids) or len(outcomes) != len(batch_ids):
                    raise RuntimeError("Batch predictions, labels and subject IDs do not align")
                ids.extend(batch_ids.tolist())
                embeddings.append(capture[0].numpy())
                probs.append(scores)
                labels.append(outcomes)
    finally:
        hook.remove()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate held-out subject IDs in prediction dataloader")
    return (np.asarray(ids, dtype=np.int64), np.concatenate(embeddings),
            np.concatenate(probs), np.concatenate(labels))


# Outcome labels are BONSAI labels for the chosen training horizon, not a
# guarantee of complete clinical follow-up when no event was recorded.
OUTCOME_RED = "#d62728"          # recorded outcome (label 1)
NO_OUTCOME_BLUE = "#1f77b4"     # no recorded outcome (label 0)


def plot_umap_scatter(coords, probabilities, labels, *, risk_max, output_dir):
    """Produce two point-only UMAP plots using identical coordinates and limits."""
    coords = np.asarray(coords, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(labels):
        raise ValueError("UMAP coordinates and outcome labels have different sizes")
    if len(probabilities) != len(coords):
        raise ValueError("UMAP coordinates and prediction probabilities have different sizes")
    if not np.isfinite(coords).all() or not np.isfinite(probabilities).all():
        raise ValueError("UMAP coordinates/predictions contain NaN or infinity")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Observed outcomes must be BONSAI binary labels (0 or 1)")
    if not 0 < risk_max <= 1:
        raise ValueError("--risk-max must be in (0, 1]")

    # Shared axis limits ensure the plots can be compared directly.
    padding = np.maximum(np.ptp(coords, axis=0) * 0.045, 0.25)
    xlim = (coords[:, 0].min() - padding[0], coords[:, 0].max() + padding[0])
    ylim = (coords[:, 1].min() - padding[1], coords[:, 1].max() + padding[1])

    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    points = ax.scatter(coords[:, 0], coords[:, 1], c=probabilities,
                        s=9, cmap="viridis", vmin=0, vmax=risk_max,
                        alpha=0.85, linewidths=0, rasterized=True)
    fig.colorbar(points, ax=ax, label="BONSAI predicted probability")
    ax.set_title("Held-out patient UMAP — predicted risk")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_dir / "umap_predicted_risk.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    # Draw the blue background cohort first, and positive outcomes on top.
    for label, color, name in (
        (0, NO_OUTCOME_BLUE, "No recorded outcome"),
        (1, OUTCOME_RED, "Recorded outcome"),
    ):
        mask = labels == label
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=9, color=color,
                       alpha=0.85, linewidths=0, rasterized=True,
                       label=f"{name} (n={int(mask.sum()):,})")
    ax.legend(loc="best", markerscale=2, frameon=False)
    ax.set_title("Held-out patient UMAP — observed outcome")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_dir / "umap_observed_outcome.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args):
    import torch
    import polars as pl
    from datetime import timedelta
    from umap import UMAP

    if args.batch_size <= 0 or args.num_workers < 0 or args.umap_neighbors < 2:
        raise ValueError("Invalid batch size, workers or UMAP n_neighbors")
    if args.umap_min_dist < 0 or not (0 < args.risk_max <= 1):
        raise ValueError("Invalid UMAP min-dist or risk color range")
    ckpt = args.checkpoint.expanduser().resolve()
    config, config_path = get_train_config(ckpt, args.finetune_config)
    root, patient_path, vocab_path, population_path, outcome_path = resolve_inputs(args, config)
    LOG.info("Dataset: %s | outcomes: %s", root, outcome_path)
    # BONSAI .pt files are pickle-serialized; load only locally trusted files.
    vocabulary = (torch.load(vocab_path, map_location="cpu", weights_only=False)
                  if vocab_path is not None else None)
    module, hp = get_model_and_module(ckpt, vocabulary, args.attn_type)
    config_max_len = config_setting(config, "training.max_len")
    max_len = args.max_len if args.max_len is not None else config_max_len
    if max_len is None:
        max_len = int(hp["max_seqlen"])
        LOG.warning("Using checkpoint max_seqlen=%s, which may not equal finetune training.max_len", max_len)
    if max_len > int(hp["max_seqlen"]) or max_len < 2:
        raise ValueError("Finetune max_len must be >=2 and <= checkpoint max_seqlen")

    start_include = config_setting(config, "labels.start_include", {"hours": 0})
    end_include = config_setting(config, "labels.end_include", None)
    from omegaconf import OmegaConf
    if OmegaConf.is_config(start_include):
        start_include = OmegaConf.to_container(start_include, resolve=True)
    if OmegaConf.is_config(end_include):
        end_include = OmegaConf.to_container(end_include, resolve=True)
    if args.label_start_hours is not None:
        start_include = {"hours": args.label_start_hours}
    if args.label_end_days is not None:
        end_include = {"days": args.label_end_days}

    cfg_horizon = timedelta(**end_include).total_seconds() / 86400 if end_include is not None else None
    if args.horizon_days is not None:
        if not np.isfinite(args.horizon_days) or args.horizon_days <= 0:
            raise ValueError("--horizon-days must be positive and finite")
        if cfg_horizon is None or abs(args.horizon_days - cfg_horizon) >= 1e-6:
            LOG.warning("--horizon-days=%s differs from label_end_days/config horizon %s. "
                        "BONSAI outcome labels still use labels.end_include; "
                        "--horizon-days does not change them.", args.horizon_days, cfg_horizon)
    if cfg_horizon is None:
        LOG.warning("No fixed outcome horizon in finetuning labels. Plots show the "
                    "training-defined outcome labels, not necessarily fixed-horizon risk.")

    loader, records, expected = make_predict_loader(
        patient_path, population_path, outcome_path, int(hp["predict_token_id"]), int(max_len),
        start_include, end_include, args.batch_size, args.num_workers)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    attention_backend = args.attn_type or hp["attn_type"]
    ids, embeddings, probabilities, labels = infer_in_one_pass(
        module, loader, device, attention_backend=attention_backend, amp_dtype=args.amp_dtype
    )
    if len(ids) != expected or any(int(sid) not in records for sid in ids):
        raise RuntimeError("Inference did not match the filtered held-out cohort")
    if not (np.isfinite(embeddings).all() and np.isfinite(probabilities).all()):
        raise ValueError("Model produced NaN/inf values")

    LOG.info("Embeddings: %d held-out patients, %d dimensions", len(ids), embeddings.shape[1])
    reducer = UMAP(n_components=2, n_neighbors=min(args.umap_neighbors, len(ids) - 1),
                   min_dist=args.umap_min_dist, random_state=args.seed,
                   transform_seed=args.seed)
    xy = reducer.fit_transform(embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_umap_scatter(xy, probabilities, labels, risk_max=args.risk_max, output_dir=args.output_dir)

    if args.save_patient_table:
        pl.DataFrame({"subject_id": ids, "umap_1": xy[:, 0], "umap_2": xy[:, 1],
                      "predicted_probability": probabilities,
                      "observed_outcome_label": labels}).write_csv(
                          args.output_dir / "patients_sensitive.csv")
    summary = {
        "checkpoint": str(ckpt), "dataset": str(root), "patient_source": str(patient_path),
        "outcomes": str(outcome_path), "vocabulary": str(vocab_path) if vocab_path else None,
        "population": str(population_path) if population_path else None,
        "original_finetune_config": str(config_path) if config_path else None,
        "n_held_out": len(ids), "embedding_dim": int(embeddings.shape[1]),
        "n_positive_labels": int(labels.sum()), "max_len": int(max_len),
        "label_start_include": start_include, "label_end_include": end_include,
        "fixed_horizon_days": cfg_horizon, "smoothed": False,
        "attention_backend": attention_backend, "requested_amp_dtype": args.amp_dtype,
        "outcome_colors": {"recorded_outcome": OUTCOME_RED,
                           "no_recorded_outcome": NO_OUTCOME_BLUE},
        "notes": ("Each point is one patient; no smoothing or censoring-adjusted visualization. "
                  "Zero means no recorded outcome in the BONSAI label window, not necessarily "
                  "confirmed event-free follow-up. If the original finetuning Hydra config is absent, "
                  "label window and max_len may differ from training."),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("Saved unsmoothed UMAP scatter plots in %s", args.output_dir.resolve())
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
