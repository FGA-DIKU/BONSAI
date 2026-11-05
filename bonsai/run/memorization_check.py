"""Supplemental greedy continuation checks on one fixed train/tuning cohort.

For each feature mode and context X, append greedy predictions to the prompt
until the longest eligible horizon, then score each shorter prefix. Compare
train/tuning means, inspect patient tails, and compare teacher-forced losses.

The primary no-patient-context run is ``bonsai.run.zero_context_check``.
Scoring and assessment are shared in ``functional.memorization_metrics``;
cohort selection, caching, audits and report figures have dedicated modules.
See docs/memorization_zero_context.md and docs/memorization_sampling.md for usage.
"""

import argparse
import hashlib
import inspect
import json
import sys
import time
import warnings
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

from bonsai.functional.memorization_cache import (
    CachedWindows,
    FixedCohortWindows,
    build_pools,
    cohort_identity,
    read_cache,
    save_cache,
    source_stamp,
)
from bonsai.functional.memorization_inspection import (
    PatientInspection,
    write_patient_losses,
)
from bonsai.functional.memorization_metrics import (
    EXCESS_COLUMNS,
    INSPECTION_HORIZON,
    LOSS_COLUMNS,
    MIN_UNIQUE_OVERLAP_TOKENS,
    MODES,
    PARTIAL_REPRODUCTION,
    SCORES,
    comparison_summary,
    count_reproduced,
    difference_in_means,
    excess_over_unseen,
    grid_coverage,
    inspection_flags,
    membership_from_loss,
    patient_flag_rates,
    patient_tails,
    score_patient,
    wilson_interval,
)
from bonsai.functional.memorization_sampling import SamplingSettings, select_cohorts
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.truncation import truncate_subject
from bonsai.modules.datasets.PretrainDataset import PretrainDataset
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiPretrain,
    BonsaiValuePretrain,
)

# Compatibility for callers importing scores from the original entry point.
__all__ = [
    "EXCESS_COLUMNS",
    "LOSS_COLUMNS",
    "count_reproduced",
    "difference_in_means",
    "inspection_flags",
    "score_patient",
    "wilson_interval",
]

FIELDS = ("code", "age", "abspos", "segment", "numeric_value")
RARE_CODE_SHARE = 0.001
PROMPTS = (32, 256, 1024)  # X; short/medium/long-context supplemental probes
CONTINUATIONS = (1, 2, 4, 8, 16, 32, 64)  # Y
FINAL_PATIENTS = 100000


def load_model(checkpoint: Path, device: str) -> BonsaiBase:
    """The pretrained model, with its next-event head, from a Lightning checkpoint."""
    saved = load_mapped(checkpoint)
    hparams = saved["hyper_parameters"]
    if not hparams.get("causal"):
        raise ValueError("The check needs a causally pretrained model (causal: true).")
    value_model = hparams.get("value_embedding_mode")
    model_class = BonsaiValuePretrain if value_model else BonsaiPretrain
    accepted = inspect.signature(model_class).parameters
    settings = {k: v for k, v in hparams.items() if k in accepted}
    settings["attention_dropout"] = 0.0  # Evaluation disables dropout.
    model = model_class(**settings)
    weights = {
        key.removeprefix("model."): value
        for key, value in saved["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)


def load_mapped(path: Path):
    """Map tensor storage instead of loading full splits/optimizer state into RAM.

    Python list/dict metadata is still loaded by torch.load. Do not silently fall
    back to an eager tensor load if a legacy file cannot be memory-mapped.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Cannot memory-map {path}. Use the modern torch.save ZIP format; "
            "an automatic full-RAM fallback is deliberately disabled."
        ) from error


class TrainingWindows(PretrainDataset):
    """Lazy pretraining windows: slice/censor before copying, never deepcopy a history.

    Two compact index arrays cache cutoff endpoints and window lengths. This lets
    sampling examine eligibility without materializing every transformed window.
    """

    def __init__(self, subjects, context, background_length, cutoff=None):
        super().__init__(subjects, context + 1, int(background_length), cutoff)
        if not 0 <= self.background_length < self.max_len:
            raise ValueError(
                "Background length must leave room for a target in the training window."
            )
        self.ends = np.fromiter(
            (
                len(row["code"])
                if self.cutoff_date is None
                else bisect_left(row["abspos"].numpy(), self.cutoff_date)
                for row in subjects
            ),
            dtype=np.int64,
            count=len(subjects),
        )
        self.lengths = np.minimum(self.ends, self.max_len)

    def _trim(self, index):
        raw, end = self.subjects[index], int(self.ends[index])
        window = {field: raw[field][:end] for field in FIELDS if field in raw}
        window["subject_id"] = raw["subject_id"]
        return truncate_subject(window, self.max_len, self.background_length)

    def prefix_codes(self, index, stop):
        """Read only code positions needed for selection, respecting training truncation."""
        end = int(self.ends[index])
        codes = self.subjects[index]["code"]
        if end <= self.max_len:
            return codes[: min(end, stop)]
        background = min(self.background_length, stop)
        tail = end - (self.max_len - self.background_length)
        return torch.cat(
            (
                codes[:background],
                codes[tail : tail + max(0, min(stop, self.max_len) - background)],
            )
        )

    def __getitem__(self, index):
        window = self._trim(index)
        # Own only bounded window tensors; nothing can mutate the mapped source.
        window = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in window.items()
        }
        window["segment"] = normalize_segments(window["segment"])
        window["attention_mask"] = torch.ones(len(window["code"]), dtype=torch.bool)
        return window


def training_windows(
    subjects, context, background_length, cutoff=None
) -> TrainingWindows:
    """Same cutoff/truncation/segments as pretraining, evaluated lazily per patient."""
    return TrainingWindows(subjects, context, background_length, cutoff)


def sample_patients(
    windows,
    min_events: int,
    max_patients: int,
    *,
    require_targets=False,
    exclude_target=None,
) -> list:
    """Uniform seeded reservoir: retain <=max_patients indices, not all eligible windows."""
    if isinstance(windows, CachedWindows):
        return windows.sample(min_events, max_patients, require_targets)
    rng, chosen, eligible = np.random.default_rng(0), [], 0
    for index in range(len(windows)):
        length = (
            int(windows.lengths[index])
            if isinstance(windows, TrainingWindows)
            else len(windows[index]["code"])
        )
        if length < min_events:
            continue
        if require_targets:
            codes = (
                windows._trim(index)["code"]
                if isinstance(windows, TrainingWindows)
                else windows[index]["code"]
            )
            valid = codes[1:] != 0
            if exclude_target is not None:
                valid &= codes[1:] != exclude_target
            if not bool(valid.any()):
                continue
        eligible += 1
        if len(chosen) < max_patients:
            chosen.append(index)
        else:
            replacement = int(rng.integers(eligible))
            if replacement < max_patients:
                chosen[replacement] = index
    return [windows[index] for index in sorted(chosen)]


def make_batch(windows: list, length: int, device) -> dict:
    """The first ``length`` events of each patient, right-padded."""
    batch = {}
    for field in FIELDS:
        if all(field in w for w in windows):
            rows = [w[field][:length] for w in windows]
            padding = float("nan") if field == "numeric_value" else 0
            batch[field] = pad_sequence(
                rows, batch_first=True, padding_value=padding
            ).to(device)
    lengths = torch.tensor([min(len(w["code"]), length) for w in windows])
    positions = torch.arange(batch["code"].shape[1])
    batch["attention_mask"] = (positions < lengths[:, None]).to(device)
    return batch


def hidden_states(model: BonsaiBase, batch: dict, code_only=False) -> torch.Tensor:
    """Causal backbone; code_only bypasses age/time/segment/value embeddings, not RoPE."""
    with torch.autocast("cuda", torch.bfloat16, enabled=batch["code"].is_cuda):
        if not code_only:
            return BonsaiBase.forward(model, batch).float()
        embedded = model.drop(model.embeddings.code_embedding(batch["code"]))
        if model.hparams["attn_type"] == "flash":
            return model.forward_flash(embedded, batch["attention_mask"]).float()
        return model.forward_sdpa(embedded, attn_mask=None).float()


@torch.inference_mode()
def generate(
    model,
    windows,
    prompt,
    steps,
    batch_size=8,
    *,
    mode="forward_fill",
    sep_token_id=None,
    supply_sep=False,
) -> np.ndarray:
    """Append greedy predictions; never feed back real continuation codes.

    code_only: use code embeddings only, retaining transformer positional encoding.
    true_features: use real continuation age/time/segment/values (an oracle stress test).
    forward_fill: hold age/time fixed; SEP closes its segment and the following
    token starts the next. Generated numeric inputs are NaN, not stale values
    copied from an unrelated code. Only true_features reads continuation features.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown generation mode: {mode}")
    if supply_sep and (mode != "true_features" or sep_token_id is None):
        raise ValueError("Supplied SEP requires true_features and a SEP token ID.")
    device = next(model.parameters()).device
    written = []
    for start in range(0, len(windows), batch_size):
        part = windows[start : start + batch_size]
        if mode == "code_only":
            features = {
                "code": torch.stack([window["code"][:prompt] for window in part]).to(
                    device
                )
            }
        else:
            features = make_batch(
                part, prompt + steps if mode == "true_features" else prompt, device
            )
        batch = {key: value[:, :prompt].clone() for key, value in features.items()}
        for position in range(prompt, prompt + steps):
            batch["attention_mask"] = torch.ones_like(batch["code"], dtype=torch.bool)
            states = (
                hidden_states(model, batch, code_only=True)
                if mode == "code_only"
                else hidden_states(model, batch)
            )
            last = states[:, -1]
            logits = model.pretrain_head(last)
            if not torch.isfinite(logits).all():
                raise ValueError(
                    "Nonfinite generation logits; no valid result can be reported."
                )
            next_code = logits.argmax(-1, keepdim=True)
            if supply_sep:
                # Only separator positions are revealed, never future clinical codes.
                is_sep = torch.tensor(
                    [int(window["code"][position]) == sep_token_id for window in part],
                    device=device,
                ).unsqueeze(1)
                next_code = torch.where(is_sep, sep_token_id, next_code)
            previous_code = batch["code"][:, -1:]
            for field in FIELDS[1:]:
                if field not in batch:
                    continue
                if mode == "true_features":
                    extension = features[field][:, position : position + 1]
                else:
                    extension = batch[field][:, -1:].clone()
                    if field == "segment" and sep_token_id is not None:
                        extension += (previous_code == sep_token_id).to(extension.dtype)
                    elif field == "numeric_value":
                        extension.fill_(float("nan"))
                batch[field] = torch.cat([batch[field], extension], dim=1)
            batch["code"] = torch.cat([batch["code"], next_code], dim=1)
        written.append(batch["code"][:, prompt:].cpu().numpy())
    return np.concatenate(written) if written else np.empty((0, steps), dtype=np.int64)


@torch.inference_mode()
def patient_losses(
    model, windows: list, batch_size: int = 8, *, exclude_target=None
) -> np.ndarray:
    """Mean next-code cross-entropy per patient; numeric-value MSE is out of scope."""
    device, losses = next(model.parameters()).device, []
    for start in range(0, len(windows), batch_size):
        part = windows[start : start + batch_size]
        batch = make_batch(part, max(len(w["code"]) for w in part) - 1, device)
        hidden = hidden_states(model, batch)
        for row, window in enumerate(part):
            targets = window["code"][1:].to(device)
            valid = targets != 0
            if exclude_target is not None:
                valid &= targets != exclude_target
            states = hidden[row, : len(targets)][valid]
            targets = targets[valid]
            if not len(targets):
                raise ValueError("Patient has no non-padding next-code targets.")
            total = 0.0
            # Bound vocabulary-logit memory: at most 512 token rows at a time.
            for start_token in range(0, len(targets), 512):
                logits = model.pretrain_head(states[start_token : start_token + 512])
                total += F.cross_entropy(
                    logits, targets[start_token : start_token + 512], reduction="sum"
                ).item()
            losses.append(total / len(targets))
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite patient losses; no valid result can be reported.")
    return np.array(losses)


def score_all_patients(
    model,
    windows_by_split,
    prompts,
    continuations,
    max_patients,
    countable,
    rare,
    batch_size=8,
    context=None,
    mode="forward_fill",
    sep_token_id=None,
    supply_sep=False,
    inspection=None,
) -> pl.DataFrame:
    """Score eligible horizons; group patients by available length to reuse decoding."""
    rows = []
    for split, windows in windows_by_split.items():
        for x in prompts:
            horizons = [
                y for y in continuations if context is None or x + y <= context + 1
            ]
            if not horizons:
                continue
            patients = sample_patients(windows, x + min(horizons), max_patients)
            # Short records still contribute to short horizons. Decode each
            # patient only as far as a requested horizon with available truth.
            groups = {}
            for window in patients:
                longest = max(y for y in horizons if x + y <= len(window["code"]))
                groups.setdefault(longest, []).append(window)
            for longest, part in sorted(groups.items()):
                print(
                    f"{mode}: {split}, X={x}, up to Y={longest}, patients={len(part)}",
                    flush=True,
                )
                generated = generate(
                    model,
                    part,
                    x,
                    longest,
                    batch_size,
                    mode=mode,
                    sep_token_id=sep_token_id,
                    **({"supply_sep": True} if supply_sep else {}),
                )
                for written, window in zip(generated, part):
                    for y in horizons:
                        if y <= longest:
                            truth = window["code"][x : x + y].numpy()
                            prediction = written[:y]
                            if supply_sep:
                                predicted_positions = truth != sep_token_id
                                truth = truth[predicted_positions]
                                prediction = prediction[predicted_positions]
                                if not len(truth):
                                    continue  # Supplied separators are not predictions.
                            metrics = score_patient(prediction, truth, countable, rare)
                            if inspection is not None:
                                inspection.record(
                                    window,
                                    written,
                                    split,
                                    mode,
                                    x,
                                    y,
                                    metrics,
                                    supply_sep,
                                    sep_token_id,
                                )
                            rows.append(
                                {
                                    "mode": mode,
                                    "split": split,
                                    "X": x,
                                    "Y": y,
                                    "sampling_cohort": window.get(
                                        "sampling_cohort", "all"
                                    ),
                                    **metrics,
                                }
                            )
    schema = {
        "mode": pl.String,
        "split": pl.String,
        "X": pl.Int64,
        "Y": pl.Int64,
        "sampling_cohort": pl.String,
        **{name: pl.Float64 for name in SCORES},
        "unique_predicted_tokens": pl.Int64,
        "unique_reference_tokens": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def parse_args():
    """Inputs, grid, sample sizes and minimal audit settings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--checkpoint", type=Path, required=True)
    add("--data-dir", type=Path, required=True)
    add("--output-dir", type=Path, required=True)
    add("--train-split", default="train")
    add("--heldout-split", default="tuning")
    add(
        "--run-config",
        type=Path,
        help="Original Hydra config; otherwise discovered above checkpoint",
    )
    add(
        "--min-unique-overlap-tokens",
        type=int,
        default=MIN_UNIQUE_OVERLAP_TOKENS,
        help="Minimum distinct predicted clinical codes for overlap flags (default 3)",
    )
    add("--data-version", help="Source dataset version for the audit record")
    add(
        "--sampling-design",
        choices=("fixed", "legacy"),
        default="fixed",
        help="fixed: one random/uncommon cohort per split; legacy: separate uniform pools per prompt/loss",
    )
    add("--random-fraction", type=float, default=0.8)
    add("--rarity-reference-patients", type=int, default=100000)
    add("--rarity-calibration-patients", type=int, default=20000)
    add("--rarity-span", type=int, default=32)
    add("--rarity-min-clinical-tokens", type=int, default=8)
    add("--rarity-quantile", type=float, default=0.9)
    add("--rarity-min-calibration", type=int, default=50)
    add("--rarity-candidate-limit", type=int, default=1000000)
    add(
        "--prepare-sampling-only",
        action="store_true",
        help="Prepare/cache fixed cohorts and write population timing/selection diagnostics; no model inference",
    )
    add(
        "--export-patient-inspection",
        action="store_true",
        help="Write local patient IDs/scores, every flagged trajectory, and unflagged examples (contains patient data)",
    )
    add(
        "--unflagged-examples-per-cell",
        type=int,
        default=3,
        help="Uniform unflagged trajectory examples per split/mode/context/horizon, seed 0",
    )
    add(
        "--evaluation-cache",
        type=Path,
        help="Local patient-window cache: create once, reuse on later runs",
    )
    add(
        "--cache-patients",
        type=int,
        default=FINAL_PATIENTS,
        help="Fixed patients per split to cache; legacy per sampling pool (default 100000)",
    )
    add(
        "--sep-excluded-from-training-loss",
        action="store_true",
        help="Exclude SEP loss targets and supply SEP positions in the conditional true_features check",
    )
    add("--context-length", type=int, help="Original training.max_len override")
    add("--population", type=Path, help="population csv used in pretraining, if any")
    add("--cutoff-date", type=date.fromisoformat, help="training.cutoff_date, if set")
    add("--prompt-events", type=int, nargs="+", default=PROMPTS)
    add("--continuation-events", type=int, nargs="+", default=CONTINUATIONS)
    add(
        "--max-patients",
        type=int,
        default=FINAL_PATIENTS,
        help="per split, fixed across the grid; legacy per prompt length",
    )
    add("--min-patients", type=int, default=50, help="per group, for summary.txt")
    add("--batch-size", type=int, default=8, help="lower it if out of GPU memory")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser, parser.parse_args()


def load_windows(data_dir, splits, context, population=None, cutoff_date=None) -> dict:
    """Map/filter one split at a time; retain source metadata and lazy windows only."""
    cohort = (
        set(pl.read_csv(population)["subject_id"].to_list()) if population else None
    )
    day = cutoff_date
    cutoff = {"year": day.year, "month": day.month, "day": day.day} if day else None
    windows, seen, background = {}, set(), None
    for split in splits:
        started = time.perf_counter()
        print(
            f"Loading {split} metadata and mapping tensor storage (full split, not sample)",
            flush=True,
        )
        records = load_mapped(data_dir / f"subject_data_{split}.pt")
        print(
            f"{split}: loaded {len(records):,} records in {time.perf_counter() - started:.1f}s; filtering population/checking IDs",
            flush=True,
        )
        if cohort is not None:
            records = [record for record in records if record["subject_id"] in cohort]
        ids = [record["subject_id"] for record in records]
        if len(records) < 2 or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(
                f"Split {split} needs distinct patients, no split overlap, and at least two records."
            )
        seen.update(ids)
        if background is None:
            background = int((records[0]["segment"] == 1).sum())
        print(
            f"{split}: indexing cutoff/window lengths for {len(records):,} retained patients",
            flush=True,
        )
        windows[split] = training_windows(records, context, background, cutoff)
        print(
            f"{split}: ready in {time.perf_counter() - started:.1f}s total", flush=True
        )
    return windows


def validate_windows(windows, vocabulary, model) -> None:
    """Fail before inference on mismatched vocabulary, ragged fields or invalid features."""
    vocab_size = model.hparams["vocab_size"]
    if (
        set(vocabulary.values()) != set(range(vocab_size))
        or len(vocabulary) != vocab_size
        or vocabulary.get("[PAD]") != 0
    ):
        raise ValueError(
            "Vocabulary must match checkpoint token-ID range and have [PAD] at 0."
        )
    required = FIELDS if model.hparams.get("value_embedding_mode") else FIELDS[:4]
    for split, records in windows.items():
        scope = "cached sample" if isinstance(records, CachedWindows) else "full split"
        print(f"Validating {split}: {len(records):,} windows ({scope})", flush=True)
        for index, window in enumerate(records):
            if index and index % 10000 == 0:
                print(f"Validating {split}: {index:,}/{len(records):,}", flush=True)
            if any(field not in window for field in required):
                raise ValueError(f"Missing required input feature in split {split}.")
            n = len(window["code"])
            if any(
                window[field].ndim != 1 or len(window[field]) != n
                for field in FIELDS
                if field in window
            ):
                raise ValueError(f"Misaligned feature lengths in split {split}.")
            if window["code"].dtype != torch.long or (
                n and (window["code"].min() < 0 or window["code"].max() >= vocab_size)
            ):
                raise ValueError(f"Invalid code IDs/dtype in split {split}.")
            if any(
                not torch.isfinite(window[field]).all() for field in ("age", "abspos")
            ):
                raise ValueError(f"Nonfinite age/time in split {split}.")
            if window["segment"].dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"Segment IDs must be integer tensors in split {split}."
                )
            if n and (
                window["segment"].min() < 0
                or window["segment"].max() >= model.hparams["max_seqlen"] + 1
            ):
                raise ValueError(f"Invalid segment IDs in split {split}.")
            if "numeric_value" in window and (
                not window["numeric_value"].is_floating_point()
                or torch.isinf(window["numeric_value"]).any()
            ):
                raise ValueError(
                    f"Numeric values must be floating point with NaN for missingness in split {split}."
                )


def code_groups(vocabulary: dict, train: list, *, return_metadata=False) -> tuple:
    """Count non-special codes; mark those held by <0.1% of training windows' patients."""
    countable = {i for token, i in vocabulary.items() if not token.startswith("[")}
    print(
        f"Counting code frequencies across {len(train):,} training windows", flush=True
    )
    held = Counter()
    for index in range(len(train)):
        # Frequencies need only censored/truncated codes, not copied features or
        # normalized segments. Avoid materializing full windows for this scan.
        codes = (
            train.prefix_codes(index, train.max_len)
            if isinstance(train, TrainingWindows)
            else train[index]["code"]
        )
        held.update(set(codes.tolist()))
        if index and index % 10000 == 0:
            print(f"Counting code frequencies: {index:,}/{len(train):,}", flush=True)
    limit = RARE_CODE_SHARE * len(train)
    rare = {code for code, n in held.items() if n < limit} & countable

    if return_metadata:
        return (
            countable,
            rare,
            {
                "patient_counts": {code: held[code] for code in rare},
                "training_patients": len(train),
                "threshold_share": RARE_CODE_SHARE,
            },
        )
    return countable, rare


def file_hash(path: Path) -> str:
    """Identify the precise artifact checked, without reading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_settings(args) -> None:
    """Recover original window settings; explicit CLI overrides take precedence."""
    args.run_config = args.run_config or next(
        (
            parent / "hydra/config.yaml"
            for parent in args.checkpoint.resolve().parents
            if (parent / "hydra/config.yaml").is_file()
        ),
        None,
    )
    if args.run_config is None:
        warnings.warn(
            "No run config found: verify population, cutoff and context manually.",
            stacklevel=2,
        )
        return
    setting = "run config file"
    try:
        cfg = OmegaConf.load(args.run_config)
        if args.population is None:
            setting = "paths.population (--population)"
            population = cfg.paths.get("population")
            args.population = Path(population) if population else None
        if args.cutoff_date is None:
            setting = "training.cutoff_date (--cutoff-date YYYY-MM-DD)"
            if cfg.training.get("cutoff_date"):
                args.cutoff_date = date(
                    **OmegaConf.to_container(cfg.training.cutoff_date, resolve=True)
                )
        if args.context_length is None:
            setting = "training.max_len (--context-length)"
            args.context_length = int(cfg.training.max_len)
    except Exception as error:
        raise ValueError(
            f"Cannot resolve original run settings from {args.run_config}: "
            f"failed at {setting}. {type(error).__name__}: {error}\n"
            "Restore the referenced training environment variables or supply the "
            "corresponding explicit override using the original training value."
        ) from error


def runtime_estimate(timings, sample_cap):
    """Separate full-dataset setup from approximately patient-scaled inference."""
    estimated = timings["setup_seconds"] + FINAL_PATIENTS / sample_cap * (
        timings["generation_scoring_seconds"] + timings["loss_seconds"]
    )
    return {
        **timings,
        "target_patients_per_group": FINAL_PATIENTS,
        "estimated_target_seconds": estimated,
        "assumptions": "Rough projection for identical grid/hardware/batch size; assumes eligible patients and representative history lengths. Excludes report writing; see coverage.csv.",
    }


def timing_clock(device):
    """Wait for outstanding CUDA work before recording wall time."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def write_outputs(
    excess,
    loss,
    output_dir: Path,
    min_patients: int,
    tails,
    manifest,
    coverage=None,
    flag_rates=None,
) -> None:
    """Write aggregate diagnostics and artifact/settings provenance; no patient records."""
    inspection_files = (
        {
            "patient_scores.csv",
            "flagged_trajectories.jsonl",
            "unflagged_examples.jsonl",
            "patient_losses.csv",
        }
        if manifest.get("patient_inspection")
        else set()
    )
    if manifest.get("sampling_design") == "fixed":
        inspection_files |= {
            "sampling_report.json",
            "sampling_patients.csv",
            "sampling_balance.csv",
        }
    if output_dir.exists() and any(
        p.name not in inspection_files for p in output_dir.iterdir()
    ):
        raise FileExistsError(
            "Use a fresh output directory so old reports cannot be mistaken for this run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    excess.write_csv(output_dir / "excess.csv")
    loss.write_csv(output_dir / "loss_check.csv")
    tails.write_csv(output_dir / "patient_tails.csv")
    if flag_rates is not None:
        flag_rates.write_csv(output_dir / "patient_flag_rates.csv")
    if coverage is not None:
        coverage.write_csv(output_dir / "coverage.csv")
    sampling_complete = manifest.get("sampling_complete", True)
    settings_complete = manifest.get("settings", {}).get("run_config") is not None
    summary = comparison_summary(
        excess,
        loss,
        min_patients,
        tails,
        coverage,
        complete=sampling_complete and settings_complete,
    )
    if manifest.get("sampling_design") == "fixed":
        summary += "\nModes retain their original assumptions; with SEP exclusion, true_features supplies future metadata and SEP positions (conditional stress test)."
        if not sampling_complete:
            summary += "\nSampling incomplete: inspect sampling_report.json; quotas/cutoffs were not weakened to fill the cohort."
        summary += "\nCombined/uncommon estimates describe enriched stress samples, not population prevalence. The random cohort supplies the population-facing comparison."
    if flag_rates is not None:
        summary += "\n\nPatient-level flag rates (per cell; not percentages memorized; do not sum across cells):"
        for row in flag_rates.filter(pl.col("inspection_applicable")).iter_rows(
            named=True
        ):
            summary += (
                f"\n  {row['mode']}/{row['split']}: X={row['X']}, Y={row['Y']}: "
                f"{row['patients_flagged']}/{row['patients']} ({row['percent_flagged']:.2f}%) flagged; "
                f"low diversity (<{row['minimum_unique_overlap_tokens']} codes): {row['patients_low_diversity']}/{row['patients']} ({row['percent_low_diversity']:.2f}%); low-diversity flags: {row['patients_flagged_low_diversity']}; unknown: {row['patients_diversity_unknown']}."
            )
            among = row["percent_low_diversity_among_exact"]
            summary += (
                f"\n    exact match = {row['patients_exact_flagged']}/{row['patients']} ({row['percent_exact_flagged']:.2f}%); "
                f"low diversity among exact = {row['patients_flagged_low_diversity']}/{row['patients_exact_flagged']} "
                f"({f'{among:.2f}%' if among is not None else 'not applicable'})."
            )
    if not settings_complete:
        summary += "\nOriginal run config unavailable: population/cutoff/context require manual verification."
    manifest["evidence_summary"] = summary.splitlines()[0]
    manifest["checks_status"] = summary.split(":", 1)[0]
    manifest["analysis_role"] = "supplemental_context_conditioned"
    manifest["generation_modes"] = {
        "code_only": "Code embeddings only; RoPE retained. Non-code embeddings bypassed.",
        "true_features": "Actual continuation age/time/segments/numeric inputs.",
        "forward_fill": "Fixed prompt age/time; segment increments after SEP; generated numeric inputs are NaN.",
    }
    if manifest.get("settings", {}).get("sep_excluded_from_training_loss", False):
        manifest["generation_modes"]["true_features"] = (
            "SUPPLEMENTAL CONDITIONAL: actual future metadata and supplied SEP positions; "
            "all other tokens greedily predicted and appended; supplied SEP not scored."
        )
    manifest["inspection_triggers"] = {
        "minimum_horizon": INSPECTION_HORIZON,
        "partial_overlap": PARTIAL_REPRODUCTION,
        "minimum_unique_predicted_clinical_tokens_for_overlap": manifest.get(
            "settings", {}
        ).get("min_unique_overlap_tokens", MIN_UNIQUE_OVERLAP_TOKENS),
        "any_exact_continuation": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    summary += "\nCheckpoint SHA-256: " + manifest["checkpoint_sha256"]
    summary += "\n" + manifest["comparison_note"]
    if manifest.get("patient_inspection"):
        summary += "\nPatient inspection (contains patient data): patient_scores.csv, flagged_trajectories.jsonl, unflagged_examples.jsonl, patient_losses.csv. Group-level excess/AUC flags do not classify specific patients as memorized."
    if timing := manifest.get("timing"):
        summary += (
            f"\n\nMeasured: setup {timing['setup_seconds'] / 60:.1f} min; "
            f"generation/scoring {timing['generation_scoring_seconds'] / 60:.1f} min; "
            f"loss {timing['loss_seconds'] / 60:.1f} min."
            f"\nEstimated {FINAL_PATIENTS:,}-patient/group run: "
            f"{timing['estimated_target_seconds'] / 3600:.2f} hours.\n"
            + timing["assumptions"]
        )
    (output_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


def fixed_settings(args):
    return SamplingSettings(
        patients_per_split=args.cache_patients
        if args.evaluation_cache
        else args.max_patients,
        random_fraction=args.random_fraction,
        reference_patients=args.rarity_reference_patients,
        calibration_patients=args.rarity_calibration_patients,
        span=args.rarity_span,
        minimum_clinical_tokens=args.rarity_min_clinical_tokens,
        quantile=args.rarity_quantile,
        minimum_calibration=args.rarity_min_calibration,
        candidate_limit=args.rarity_candidate_limit,
    )


def build_fixed_windows(windows, args, context, countable, excluded_target, vocab_size):
    settings = fixed_settings(args)
    prompts = [x for x in args.prompt_events if x + settings.span <= context + 1]
    if not prompts:
        raise ValueError(
            "No rarity span fits the model window; reduce --rarity-span or adjust prompts."
        )

    def read_codes(split, index, stop):
        codes = windows[split].prefix_codes(index, stop)
        if (
            codes.ndim != 1
            or codes.dtype != torch.long
            or (len(codes) and (codes.min() < 0 or codes.max() >= vocab_size))
        ):
            raise ValueError(
                "Invalid code IDs/dtype in a rarity reference or candidate window."
            )
        return codes.numpy()

    minimum_length = min(
        x + y
        for x in args.prompt_events
        for y in args.continuation_events
        if x + y <= context + 1
    )
    selected, audit = select_cohorts(
        {split: rows.lengths for split, rows in windows.items()},
        read_codes,
        countable,
        prompts,
        minimum_length,
        settings,
        excluded_target,
    )
    pools = {}
    for split, members in selected.items():
        rows = []
        for member in members:
            row = windows[split][member["index"]]
            row["sampling_cohort"] = member["cohort"]
            rows.append(row)
            member["subject_id"] = row["subject_id"]
            member["window_length"] = len(row["code"])
        pools[split] = {
            "rows": rows,
            "members": members,
            "random_fraction": settings.random_fraction,
            "excluded_target": excluded_target,
        }
    return pools, audit


def write_sampling_overview(output_dir, windows, audit, maximum):
    """Local ID-level selection provenance plus aggregate balance/coverage checks."""
    overview = {
        key: value
        for key, value in audit.items()
        if key not in ("reference_indices", "calibration_indices")
    }
    overview["reference_patients_reserved"] = len(audit["reference_indices"])
    overview["calibration_patients_reserved"] = len(audit["calibration_indices"])
    (output_dir / "sampling_report.json").write_text(
        json.dumps(overview, indent=2) + "\n"
    )
    records = []
    for split, dataset in windows.items():
        selected_ids = {row["subject_id"] for row in dataset.sample(0, maximum, False)}
        for member in dataset.members:
            if member["subject_id"] in selected_ids:
                records.append(
                    {
                        "split": split,
                        **{k: v for k, v in member.items() if k != "index"},
                    }
                )
    frame = pl.DataFrame(records)
    frame.write_csv(output_dir / "sampling_patients.csv")
    balance = frame.group_by("split", "cohort", "rarity_mask", "rarity_stratum").agg(
        pl.len().alias("patients"),
        pl.col("rarity_score").median().alias("median_surprise"),
        pl.col("rarity_score").quantile(0.1).alias("p10_surprise"),
        pl.col("rarity_score").quantile(0.9).alias("p90_surprise"),
        pl.col("window_length").median().alias("median_window_length"),
    )
    balance.sort("split", "cohort", "rarity_mask", "rarity_stratum").write_csv(
        output_dir / "sampling_balance.csv"
    )


def write_cohort_reports(args, scores, patients, losses, manifest, context):
    """Reuse inference for random, uncommon and combined reports; no extra decoding."""
    splits = (args.train_split, args.heldout_split)
    statuses = {"combined": manifest["checks_status"]}
    for cohort in ("random", "uncommon"):
        part = scores.filter(pl.col("sampling_cohort") == cohort)
        arrays = [
            np.array(
                [
                    value
                    for row, value in zip(records, values)
                    if row["sampling_cohort"] == cohort
                ]
            )
            for records, values in zip(patients, losses)
        ]
        subset_manifest = json.loads(json.dumps(manifest, default=str))
        subset_manifest["analysis_cohort"] = cohort
        subset_manifest["patient_inspection"] = None
        subset_manifest.pop("timing", None)
        subset_manifest["reanalysis"] = {
            **manifest.get("reanalysis", {}),
            "source_trajectory_exports": manifest.get("reanalysis", {}).get(
                "source_trajectory_exports", str(args.output_dir.resolve())
            ),
        }
        if cohort == "random":
            subset_manifest["sampling_complete"] = True
        destination = args.output_dir / "cohorts" / cohort
        subset_manifest["settings"]["output_dir"] = str(destination)
        coverage = grid_coverage(
            part,
            args.prompt_events,
            args.continuation_events,
            splits,
            context,
            args.min_patients,
            minimum_diversity=max(
                (
                    f["minimum_unique_tokens"]
                    for f in subset_manifest.get(
                        "diversity_filters",
                        [subset_manifest.get("diversity_filter", {})],
                    )
                    if "minimum_unique_tokens" in f
                ),
                default=None,
            ),
        )
        write_outputs(
            excess_over_unseen(part, *splits),
            membership_from_loss(*arrays, allow_insufficient=True),
            destination,
            args.min_patients,
            patient_tails(part, args.min_unique_overlap_tokens),
            subset_manifest,
            coverage,
            patient_flag_rates(part, args.min_unique_overlap_tokens),
        )
        statuses[cohort] = subset_manifest["checks_status"]
        if args.export_patient_inspection:
            for name in ("patient_scores.csv", "patient_losses.csv"):
                pl.scan_csv(args.output_dir / name).filter(
                    pl.col("sampling_cohort") == cohort
                ).sink_csv(destination / name)
    if "CHECK" in statuses.values():
        overall = "CHECK: at least one random, uncommon or combined comparison requires investigation."
    elif "INCONCLUSIVE" in statuses.values():
        overall = "INCONCLUSIVE: some random, uncommon or combined comparisons lack adequate evidence."
    else:
        overall = "CHECKS PASSED: all required random, uncommon and combined comparisons include their nulls."
    manifest["cohort_statuses"] = statuses
    manifest["checks_status"] = overall.split(":", 1)[0]
    manifest["evidence_summary"] = overall
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    path = args.output_dir / "summary.txt"
    old = path.read_text()
    path.write_text(
        overall
        + "\n"
        + "Cohort statuses: "
        + ", ".join(f"{name}={status}" for name, status in statuses.items())
        + "\nCombined-sample details follow:\n"
        + old.partition("\n")[2]
    )
    print(overall, flush=True)


def main() -> None:
    started = time.perf_counter()
    load_dotenv()  # Match the training entry point when Hydra references .env paths.
    parser, args = parse_args()
    if args.prepare_sampling_only and args.sampling_design != "fixed":
        parser.error("--prepare-sampling-only requires --sampling-design fixed.")
    if args.sampling_design == "fixed":
        try:
            fixed_settings(args).validate()
        except ValueError as error:
            parser.error(str(error))
    if args.min_unique_overlap_tokens < 1:
        parser.error("--min-unique-overlap-tokens must be positive.")
    if args.unflagged_examples_per_cell < 0:
        parser.error("--unflagged-examples-per-cell must be nonnegative.")

    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error(
            "Use a new or empty output directory; existing reports are not overwritten."
        )
    if args.max_patients < 2 or args.min_patients < 2:
        parser.error(
            "max-patients and min-patients must be at least 2 for the loss comparison."
        )
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is not available.")
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                parser.error(
                    "This check uses BF16 on CUDA; the selected GPU must support BF16."
                )
    if args.train_split == args.heldout_split:
        parser.error("Train and comparison splits must differ.")
    if (
        min(
            args.batch_size,
            args.max_patients,
            args.min_patients,
            *args.prompt_events,
            *args.continuation_events,
        )
        < 1
    ):
        parser.error("Batch/sample counts and prompt/horizon lengths must be positive.")
    args.prompt_events = sorted(set(args.prompt_events))
    args.continuation_events = sorted(set(args.continuation_events))
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        training_settings(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    # 1. Restore the causal model and the windows used by pretraining.
    checkpoint_sha256 = file_hash(args.checkpoint)
    model = load_model(args.checkpoint, args.device)
    context = (
        args.context_length
        if args.context_length is not None
        else model.hparams["max_seqlen"]
    )
    if not 1 <= context <= model.hparams["max_seqlen"]:
        parser.error(
            "Training context must be positive and within checkpoint max_seqlen."
        )
    if not any(
        x + y <= context + 1
        for x in args.prompt_events
        for y in args.continuation_events
    ):
        parser.error("No requested context/horizon fits the model window.")
    splits = (args.train_split, args.heldout_split)
    vocabulary = torch.load(args.data_dir / "vocabulary.pt", weights_only=False)
    validate_windows({}, vocabulary, model)
    sep_id = vocabulary.get("[SEP]")
    if args.sep_excluded_from_training_loss and sep_id is None:
        parser.error("--sep-excluded-from-training-loss requires [SEP] in vocabulary.")
    excluded_target = sep_id if args.sep_excluded_from_training_loss else None
    cache_info = None
    rare_metadata = None
    sampling_audit = None
    signature = None
    if args.evaluation_cache:
        if args.cache_patients < args.max_patients:
            parser.error("--cache-patients must be at least --max-patients.")
        signature = {
            "sources": [
                source_stamp(args.data_dir / f"subject_data_{split}.pt")
                for split in splits
            ],
            "population_sha256": file_hash(args.population)
            if args.population
            else None,
            "vocabulary_sha256": file_hash(args.data_dir / "vocabulary.pt"),
            "context": context,
            "cutoff": str(args.cutoff_date),
            "prompts": args.prompt_events,
            "horizons": args.continuation_events,
            "excluded_target": excluded_target,
            "data_version": args.data_version,
            "seed": 0,
            "rare_code_share": RARE_CODE_SHARE,
        }
    if signature is not None and args.sampling_design == "fixed":
        signature["sampling_design"] = "fixed_v1"
        signature["sampling_settings"] = asdict(fixed_settings(args))
    if args.evaluation_cache and args.evaluation_cache.exists():
        print(f"Loading sampled evaluation cache: {args.evaluation_cache}", flush=True)
        windows, countable, rare, capacity = read_cache(
            args.evaluation_cache, signature, args.max_patients
        )
        validate_windows(windows, vocabulary, model)
        sampling_audit = next(iter(windows.values())).sampling_audit
        rare_metadata = next(iter(windows.values())).rare_metadata
        cache_info = {
            "path": str(args.evaluation_cache),
            "reused": True,
            "capacity": capacity,
            "signature": signature,
        }
    else:
        load_started = time.perf_counter()
        source_windows = load_windows(
            args.data_dir, splits, context, args.population, args.cutoff_date
        )
        load_seconds = time.perf_counter() - load_started
        if args.sampling_design == "fixed":
            countable = {
                i for token, i in vocabulary.items() if not token.startswith("[")
            }
            pools, sampling_audit = build_fixed_windows(
                source_windows,
                args,
                context,
                countable,
                excluded_target,
                model.hparams["vocab_size"],
            )
            sampling_audit["population_load_index_seconds"] = load_seconds
            windows = {split: FixedCohortWindows(pool) for split, pool in pools.items()}
            validate_windows(windows, vocabulary, model)
        else:
            windows = source_windows
            validate_windows(windows, vocabulary, model)
            pools = None
        frequency_started = time.perf_counter()
        countable, rare, rare_metadata = code_groups(
            vocabulary, source_windows[args.train_split], return_metadata=True
        )
        if sampling_audit is not None:
            sampling_audit["global_rare_code_metric_scan_seconds"] = (
                time.perf_counter() - frequency_started
            )
        if args.evaluation_cache:
            if pools is None:
                minimum_lengths = sorted(
                    {
                        x
                        + min(
                            y for y in args.continuation_events if x + y <= context + 1
                        )
                        for x in args.prompt_events
                        if any(x + y <= context + 1 for y in args.continuation_events)
                    }
                )
                pools = build_pools(
                    source_windows,
                    minimum_lengths,
                    args.cache_patients,
                    sample_patients,
                    excluded_target,
                    generation_limit=max(
                        x + y
                        for x in args.prompt_events
                        for y in args.continuation_events
                        if x + y <= context + 1
                    ),
                )
            if signature["sources"] != [
                source_stamp(args.data_dir / f"subject_data_{split}.pt")
                for split in splits
            ]:
                raise ValueError(
                    "Source files changed during evaluation-cache preparation."
                )
            save_cache(
                args.evaluation_cache,
                signature,
                args.cache_patients,
                pools,
                countable,
                rare,
                sampling_audit,
                rare_metadata=rare_metadata,
            )
            windows = {
                split: (
                    FixedCohortWindows(pool)
                    if sampling_audit is not None
                    else CachedWindows(pool)
                )
                for split, pool in pools.items()
            }
            cache_info = {
                "path": str(args.evaluation_cache),
                "reused": False,
                "capacity": args.cache_patients,
                "signature": signature,
            }
            print(f"Evaluation cache saved: {args.evaluation_cache}", flush=True)
        del source_windows
    if sampling_audit is not None:
        write_sampling_overview(
            args.output_dir, windows, sampling_audit, args.max_patients
        )
    selected_identity = (
        cohort_identity(
            {
                split: dataset.sample(0, args.max_patients, False)
                for split, dataset in windows.items()
            }
        )
        if args.sampling_design == "fixed"
        else None
    )
    if args.prepare_sampling_only:
        preparation = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "vocabulary_sha256": file_hash(args.data_dir / "vocabulary.pt"),
            "settings": vars(args),
            "evaluation_cache": cache_info,
            "effective_context": context,
            "sampling_design": args.sampling_design,
            "sampling_complete": sampling_audit is not None
            and sampling_audit["calibration_reliable"]
            and all(part["complete"] for part in sampling_audit["splits"].values()),
            "analysis_role": "sampling_preparation_only",
            "cohort_id_sha256": selected_identity,
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(preparation, indent=2, default=str) + "\n"
        )
        (args.output_dir / "sampling_preparation.txt").write_text(
            f"Sampling preparation only; no model inference. Total setup: {time.perf_counter() - started:.1f} seconds.\n"
            "See sampling_report.json for source-loading, reference/calibration/candidate, and global rare-code scan timings.\n"
            "See sampling_patients.csv and sampling_balance.csv for cohorts and balance. This is not a memorization verdict.\n"
        )
        print(
            f"Sampling preparation complete; no inference. Reports: {args.output_dir}",
            flush=True,
        )
        return
    setup_done = timing_clock(args.device)

    # 2. Greedily continue codes over the X/Y grid; compare training vs unseen patients.
    inspection = (
        PatientInspection(
            args.output_dir,
            vocabulary,
            args.unflagged_examples_per_cell,
            INSPECTION_HORIZON,
            PARTIAL_REPRODUCTION,
            minimum_unique_overlap=args.min_unique_overlap_tokens,
        )
        if args.export_patient_inspection
        else None
    )
    scores = pl.concat(
        [
            score_all_patients(
                model,
                windows,
                args.prompt_events,
                sorted(args.continuation_events),
                args.max_patients,
                countable,
                rare,
                args.batch_size,
                context=context,
                mode=mode,
                sep_token_id=vocabulary.get("[SEP]"),
                supply_sep=args.sep_excluded_from_training_loss
                and mode == "true_features",
                inspection=inspection,
            )
            for mode in MODES
        ]
    )
    if inspection is not None:
        inspection.close()
    excess = excess_over_unseen(scores, *splits)
    coverage = grid_coverage(
        scores,
        args.prompt_events,
        args.continuation_events,
        splits,
        context,
        args.min_patients,
    )

    generation_done = timing_clock(args.device)

    # 3. Independently compare the groups' next-code losses (membership AUC).
    patients = [
        sample_patients(
            windows[s],
            2,
            args.max_patients,
            require_targets=True,
            exclude_target=excluded_target,
        )
        for s in splits
    ]
    losses = []
    for split, part in zip(splits, patients):
        print(f"Teacher-forced loss: {split}, patients={len(part)}", flush=True)
        losses.append(
            patient_losses(model, part, args.batch_size, exclude_target=excluded_target)
        )
    loss = membership_from_loss(
        *losses, allow_insufficient=args.sampling_design == "fixed"
    )
    if inspection is not None:
        write_patient_losses(args.output_dir, splits, patients, losses)
    loss_done = timing_clock(args.device)

    # 4. Export aggregate diagnostics, never an automatic sharing approval.
    if file_hash(args.checkpoint) != checkpoint_sha256:
        raise ValueError(
            "Checkpoint changed during the check; rerun on the final artifact."
        )
    manifest = {
        "checkpoint": args.checkpoint,
        "cohort_id_sha256": selected_identity,
        "checkpoint_sha256": checkpoint_sha256,
        "run_config_sha256": file_hash(args.run_config) if args.run_config else None,
        "settings": vars(args),
        "effective_context": context,
        "evaluation_cache": cache_info,
        "sampling_design": args.sampling_design,
        "sampling_complete": all(
            part["complete"] for part in sampling_audit["splits"].values()
        )
        and sampling_audit["calibration_reliable"]
        if sampling_audit is not None
        else True,
        "patient_inspection": {
            "unflagged_examples_per_cell": args.unflagged_examples_per_cell,
            "seed": 0,
            "scope": "evaluated windows, not complete lifetime histories",
            "flags": "same per-patient exact/overlap triggers as patient_tails.csv; not individual memorization labels",
        }
        if args.export_patient_inspection
        else None,
        "separator_policy": {
            "excluded_from_training_loss": args.sep_excluded_from_training_loss,
            "loss_excluded_target_id": excluded_target,
            "true_features_supplies_sep": args.sep_excluded_from_training_loss,
            "horizon_units": "sequence positions including SEP",
        },
        "timing": runtime_estimate(
            {
                "setup_seconds": setup_done - started,
                "generation_scoring_seconds": generation_done - setup_done,
                "loss_seconds": loss_done - generation_done,
            },
            args.max_patients,
        ),
        "vocabulary_sha256": file_hash(args.data_dir / "vocabulary.pt"),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "polars": pl.__version__,
            "numpy": np.__version__,
        },
        "sampling_seed": 0,
        "sampling_method": "uniform reservoir pools; nested seeded subsets when cache enabled"
        if cache_info
        else "uniform reservoir over eligible patient indices",
        "host_memory": "sampled bounded windows from evaluation cache"
        if cache_info
        else "memory-mapped tensor storage; lazy bounded windows; full Python metadata still loaded",
        "precision": "bf16 backbone / fp32 logits"
        if str(args.device).startswith("cuda")
        else "fp32",
        "data_version": args.data_version or "not supplied",
        "release_threshold": None,
        "status": "EXPLORATORY_ONLY",
        "comparison_note": "Tuning patients may influence checkpoint selection; this is a validation comparison."
        if args.heldout_split == "tuning"
        else "Verify comparison patients were not used for training/selection.",
    }
    write_outputs(
        excess,
        loss,
        args.output_dir,
        args.min_patients,
        patient_tails(scores, args.min_unique_overlap_tokens),
        manifest,
        coverage,
        patient_flag_rates(scores, args.min_unique_overlap_tokens),
    )
    if args.sampling_design == "fixed":
        write_cohort_reports(args, scores, patients, losses, manifest, context)
        with (args.output_dir / "summary.txt").open("a") as stream:
            stream.write(
                "\nCohort comparisons: this directory is the combined sample; cohorts/random/ and cohorts/uncommon/ contain separate reports using the same inference.\n"
            )
    if inspection is not None:
        from bonsai.run.inspect_memorization import write_audit

        print("Writing vim-ready patient audit: audit.txt", flush=True)
        write_audit(
            args.output_dir,
            args.output_dir / "audit.txt",
            args.min_unique_overlap_tokens,
        )
        with (args.output_dir / "summary.txt").open("a") as stream:
            stream.write(
                "\nOpen audit.txt in vim for indexed patient cases, comparisons, histories and review notes.\n"
            )
        print(f"Patient audit ready: {args.output_dir / 'audit.txt'}", flush=True)

    from bonsai.run.report_memorization import write_report

    print(f"Readable report ready: {write_report(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
