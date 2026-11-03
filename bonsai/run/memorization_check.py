"""Memorization check: does a pretrained BONSAI model reproduce its training patients?

The model gets the first X events of a patient's record and continues it for Y
events, always taking its most likely next code. Its continuation is compared
with the real record:

    exact_match  all Y events reproduced, in order
    overlap      share of the next Y real events also written, in any order
    rare_codes   share of the patient's rare codes (held by < 0.1% of training
                 patients) also written

Three generation modes probe different feature assumptions: code_only removes
non-code embeddings; true_features supplies real continuation features; forward_fill
holds age/time fixed, advances segments after [SEP], and leaves generated numeric
inputs missing. No one convention is definitive. CSV rows identify each mode.

For checkpoints trained without SEP loss, add --sep-excluded-from-training-loss
to BOTH preflight and main commands below. The primary true_features run then
supplies real SEP positions, predicts/appends all other tokens, and excludes
supplied SEP from scores. Loss also excludes SEP targets. The other modes keep
free generation as supplementary checks. X/Y remain sequence-position counts,
including separators, rather than counts of clinical tokens. This conditional
check reveals future structure; it is not unrestricted extraction.

Records are partly predictable, so each score is compared with patients
not used for gradient updates. An excess is a diagnostic signal, not proof of
memorization. Tuning patients may have influenced checkpoint selection.
The next-event loss of the two groups is compared too: an AUC of 0.5 means a
patient's loss has no average ranking advantage for training membership.

Windows follow pretraining's transforms using the recovered training settings.
patient_flag_rates.csv reports flagged/evaluated patients and percentage per
split/mode/context/horizon, counting each patient once even if both triggers
fire. Short horizons below the inspection threshold have no flag percentage.
patient_tails.csv also includes percentage columns for each reproduction metric.
Outputs are ``excess.csv``, ``loss_check.csv``, ``patient_tails.csv``,
``coverage.csv``, ``run_manifest.json`` and ``summary.txt``. Individual records
and scores are not exported in reports. The optional local cache contains patient
windows and must stay on the data server. The summary is for human review.
To inspect individual cases, add --export-patient-inspection. This additionally
writes patient_scores.csv (every scored ID/cell, metrics, flags),
flagged_trajectories.jsonl (every patient-level exact/overlap trigger),
unflagged_examples.jsonl (seeded sample of up to 3 unflagged examples per cell;
change with --unflagged-examples-per-cell), and patient_losses.csv. These files
contain patient data and must stay on the server. Trajectories include decoded
prompt, reference, prediction, true features and supplied SEP positions (zero-based
offsets within the continuation). History is the evaluated window, not the full
lifetime record. "Unflagged" means no configured reproduction trigger, not safe
or non-memorized; short horizons are below the inspection threshold. Group-level
excess/AUC signals do not identify particular patients as memorized.
Existing reports have no stored predictions: rerun with the same cache/settings
and this flag in a fresh output directory to obtain inspection exports.
Read a saved example with python -m bonsai.run.inspect_memorization
--report-dir memorization --subject-id 123 (optional --mode/--context/--horizon).
Optional plots after completion (no inference): install with pip install '.[plots]'
then python -m bonsai.run.plot_memorization --report-dir memorization.
A pass requires enough patients in every requested
context/horizon cell in all three modes; unavailable cells are explicit.
A single inspection trigger still warrants checking even with few patients.
Tensor storage is memory-mapped and transformed windows are lazy. Python metadata
for each split still occupies RAM; max-patients caps retained inference samples,
not that metadata. Global validation/rarity scans do not retain all windows.

Default run: up to 5,000 patients per group, contexts 32/256/1024, horizons
1/2/4/8/16/32/64, all three modes. Shorter horizons reuse the longest decoded
continuation. DATA_DIR contains subject_data_{train,tuning}.pt and vocabulary.pt.
Use the original run config and a fresh output directory for each command.
Add --evaluation-cache /local/path/memorization_cache.pt to both commands to
prepare/reuse sampled windows. The first run still scans the source splits once
and prepares up to --cache-patients (default 5000) per context/loss pool, even
when --max-patients is 100. Later runs load only cached windows and rarity groups.
Sources (path/size/mtime), population/vocabulary hashes and window/grid/SEP
settings must match; use a new cache path when they change. Cache writes are
atomic. Seeded subsets are nested within each cached pool. Reports record reuse.

Timing preflight: 100 patients per group, the full grid and all modes. Use the
same GPU and batch size as the main run. This checks the largest cell too.
Check coverage.csv: short histories can reduce the actual sample in each cell.
The preflight is a timing/execution check, not the final statistical assessment.

    python -m bonsai.run.memorization_check --checkpoint best.ckpt \\
        --run-config hydra/config.yaml --data-dir DATA_DIR \\
        --output-dir memorization_preflight --max-patients 100 --batch-size 32 \\
        --evaluation-cache /local/path/memorization_cache.pt

summary.txt and run_manifest.json record setup, generation/scoring and loss
timings, plus a rough 5,000-patient estimate: setup + 50 * (generation + loss)
for a 100-patient preflight. Setup includes full-data validation/rarity scans.
With a newly created cache this projection includes one-time cache preparation;
later runs reuse it and should have a smaller setup cost.
The estimate excludes report writing and assumes enough eligible patients,
similar history lengths, identical hardware/grid/batch size and steady throughput.
Partial batches, GPU warm-up and repeated sampling scans can overestimate cost;
different loss-window lengths or resource contention can shift it either way.

Main run (use the batch size verified above; 32 is an example, default is 8):

    python -m bonsai.run.memorization_check --checkpoint best.ckpt \\
        --run-config hydra/config.yaml --data-dir DATA_DIR \\
        --output-dir memorization --batch-size 32 \\
        --evaluation-cache /local/path/memorization_cache.pt
"""

# Flow: load windows -> generate/score the X/Y grid -> compare group means ->
# compare next-code losses -> write aggregate results. main() shows these stages.
# Each mode makes a different feature assumption; inspect a finding in any mode.

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
import warnings
from bisect import bisect_left
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from bonsai.functional.memorization_cache import (
    CachedWindows,
    build_pools,
    read_cache,
    save_cache,
    source_stamp,
)
from bonsai.functional.memorization_inspection import (
    PatientInspection,
    write_patient_losses,
)
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.truncation import truncate_subject
from bonsai.modules.datasets.PretrainDataset import PretrainDataset
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiPretrain,
    BonsaiValuePretrain,
)

FIELDS = ("code", "age", "abspos", "segment", "numeric_value")
MODES = ("code_only", "true_features", "forward_fill")
SCORES = ("exact_match", "overlap", "rare_codes")
RARE_CODE_SHARE = 0.001
# Conservative inspection triggers, not thresholds for approving model sharing.
INSPECTION_HORIZON = 32
PARTIAL_REPRODUCTION = 0.9
PROMPTS = (32, 256, 1024)  # X; primary check plus medium/long-context probes
CONTINUATIONS = (1, 2, 4, 8, 16, 32, 64)  # Y
FINAL_PATIENTS = 5000
LOSS_COLUMNS = [
    "patients_train",
    "patients_unseen",
    "loss_train",
    "loss_unseen",
    "auc",
    "auc_lower",
    "auc_upper",
]
EXCESS_COLUMNS = [
    "mode",
    "score",
    "X",
    "Y",
    "patients_train",
    "patients_unseen",
    "train",
    "unseen",
    "excess",
    "lower",
    "upper",
]


# -- 1. Model and training windows ---------------------------------------------------------------


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


# -- 2. Model inference: greedy continuation and teacher-forced loss ----------------------------------------------------------


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


# -- 3. Per-patient reproduction scores -----------------------------------------------------------------------


def count_reproduced(generated, truth, codes: set):
    """(real events reproduced, real events counted), over real codes in ``codes``.

    Order does not matter, but each written code stands in for one real event
    only.
    """
    used = np.zeros(len(generated), dtype=bool)
    reproduced = counted = 0
    for code in truth:
        if code not in codes:
            continue
        counted += 1
        matches = (generated == code) & ~used
        if matches.any():
            used[matches.argmax()] = True
            reproduced += 1
    return reproduced, counted


def score_patient(generated, truth, countable: set, rare: set) -> dict:
    """One patient's three scores (NaN when there is nothing to count)."""
    hits, n = count_reproduced(generated, truth, countable)
    rare_hits, rare_n = count_reproduced(generated, truth, rare)
    return {
        "exact_match": float((generated == truth).all()) if len(truth) else math.nan,
        "overlap": hits / n if n else math.nan,
        "rare_codes": rare_hits / rare_n if rare_n else math.nan,
    }


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
                                    **metrics,
                                }
                            )
    schema = {
        "mode": pl.String,
        "split": pl.String,
        "X": pl.Int64,
        "Y": pl.Int64,
        **{name: pl.Float64 for name in SCORES},
    }
    return pl.DataFrame(rows, schema=schema)


def grid_coverage(scores, prompts, horizons, splits, context, min_patients):
    """Every requested cell, including unsupported/undersampled cells; no silent skips."""
    counts = {
        (row["mode"], row["split"], row["X"], row["Y"]): row["len"]
        for row in scores.group_by("mode", "split", "X", "Y")
        .len()
        .iter_rows(named=True)
    }
    rows = []
    for mode in MODES:
        for split in splits:
            for x in sorted(set(prompts)):
                for y in sorted(set(horizons)):
                    n = counts.get((mode, split, x, y), 0)
                    status = (
                        "unsupported_context"
                        if x + y > context + 1
                        else "too_few_patients"
                        if n < min_patients
                        else "complete"
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "split": split,
                            "X": x,
                            "Y": y,
                            "patients": n,
                            "status": status,
                        }
                    )
    return pl.DataFrame(rows)


def patient_tails(scores: pl.DataFrame) -> pl.DataFrame:
    """Describe per-patient tails without exporting individual scores or identifiers.

    p95/max and counts scoring 1 expose concentrated reproduction hidden by means.
    These are descriptive summaries, not a calibrated membership attack or gate.
    """
    values = scores.unpivot(
        on=SCORES,
        index=["mode", "split", "X", "Y"],
        variable_name="score",
        value_name="value",
    ).filter(pl.col("value").is_finite())
    return (
        values.group_by("mode", "split", "X", "Y", "score")
        .agg(
            pl.len().alias("patients"),
            pl.col("value").median().alias("median"),
            pl.col("value").quantile(0.95, interpolation="linear").alias("p95"),
            pl.col("value").max().alias("max"),
            (pl.col("value") == 1).sum().alias("patients_scoring_1"),
            (pl.col("value") >= PARTIAL_REPRODUCTION)
            .sum()
            .alias("patients_scoring_ge_90pct"),
        )
        .with_columns(
            (100 * pl.col("patients_scoring_1") / pl.col("patients")).alias(
                "percent_scoring_1"
            ),
            (100 * pl.col("patients_scoring_ge_90pct") / pl.col("patients")).alias(
                "percent_scoring_ge_90pct"
            ),
        )
        .sort("mode", "split", "X", "Y", "score")
    )


def patient_flag_rates(scores):
    """Per-cell union of the patient-level triggers: never count one patient twice.

    Denominator is evaluated patients, not the full cohort. Below the inspection
    horizon flags are not assessed; percentages are null, rather than reassuring 0.
    """
    applicable = pl.col("Y") >= INSPECTION_HORIZON
    exact = (pl.col("exact_match") == 1).fill_null(False)
    overlap = (
        pl.col("overlap").is_finite() & (pl.col("overlap") >= PARTIAL_REPRODUCTION)
    ).fill_null(False)
    return (
        scores.group_by("mode", "split", "X", "Y")
        .agg(
            pl.len().alias("patients"),
            ((exact | overlap) & applicable).sum().alias("patients_flagged"),
            (exact & applicable).sum().alias("patients_exact_flagged"),
            (overlap & applicable).sum().alias("patients_overlap_flagged"),
        )
        .with_columns(
            applicable.alias("inspection_applicable"),
            pl.when(applicable)
            .then(100 * pl.col("patients_flagged") / pl.col("patients"))
            .otherwise(None)
            .alias("percent_flagged"),
        )
        .sort("mode", "split", "X", "Y")
    )


# -- 4. Training-versus-unseen aggregate statistics -------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion; not empty when no patient matches."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def difference_in_means(train: np.ndarray, unseen: np.ndarray, is_proportion: bool):
    """Mean of ``train`` minus mean of ``unseen`` with a 95% interval: Newcombe
    (1998, method 10) for exact match (yes/no), otherwise normal (Welch)."""
    p1, p2 = train.mean(), unseen.mean()
    if is_proportion:
        l1, u1 = wilson_interval(int(train.sum()), len(train))
        l2, u2 = wilson_interval(int(unseen.sum()), len(unseen))
        below, above = math.hypot(p1 - l1, u2 - p2), math.hypot(u1 - p1, p2 - l2)
        return p1 - p2, p1 - p2 - below, p1 - p2 + above
    variance = train.var(ddof=1) / len(train) + unseen.var(ddof=1) / len(unseen)
    half = 1.96 * math.sqrt(variance)
    return p1 - p2, p1 - p2 - half, p1 - p2 + half


def excess_over_unseen(
    scores: pl.DataFrame, train_split: str, unseen_split: str
) -> pl.DataFrame:
    """Per score, X and Y: both groups' means and the excess, counting only the
    patients who have the score (e.g. rare_codes needs a rare code in the next Y)."""
    rows = []
    for (mode, x, y), part in scores.group_by("mode", "X", "Y"):
        in_train = part.filter(pl.col("split") == train_split)
        in_unseen = part.filter(pl.col("split") == unseen_split)
        for name in SCORES:
            train = in_train[name].drop_nans().to_numpy()
            unseen = in_unseen[name].drop_nans().to_numpy()
            if len(train) < 2 or len(unseen) < 2:
                continue
            is_proportion = name == "exact_match"
            excess = difference_in_means(train, unseen, is_proportion)
            counts = len(train), len(unseen), train.mean(), unseen.mean()
            rows.append((mode, name, x, y, *counts, *excess))
    excess = pl.DataFrame(rows, schema=EXCESS_COLUMNS, orient="row")
    return excess.sort("mode", "score", "X", "Y")


def membership_from_loss(train: np.ndarray, unseen: np.ndarray) -> pl.DataFrame:
    """AUC for telling training from unseen patients by their loss (0.5 = not at
    all), with the Hanley and McNeil (1982) 95% interval."""
    n1, n2 = len(train), len(unseen)
    if min(n1, n2) < 2 or not np.isfinite(train).all() or not np.isfinite(unseen).all():
        raise ValueError(
            "Loss comparison needs at least two finite patient losses per group."
        )
    auc = roc_auc_score([1] * n1 + [0] * n2, -np.concatenate([train, unseen]))
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)
    half = 1.96 * math.sqrt(max(0.0, var) / (n1 * n2))
    row = (
        n1,
        n2,
        train.mean(),
        unseen.mean(),
        auc,
        max(0.0, auc - half),
        min(1.0, auc + half),
    )
    return pl.DataFrame([row], schema=LOSS_COLUMNS, orient="row")


# -- 5. Interpretation and text summary -------------------------------------------


def inspection_flags(tails: pl.DataFrame) -> pl.DataFrame:
    """Flag any exact or >=90% unordered reproduction at horizons >=32, even at n=1."""
    return tails.filter(
        (pl.col("Y") >= INSPECTION_HORIZON)
        & (
            ((pl.col("score") == "exact_match") & (pl.col("patients_scoring_1") > 0))
            | (
                (pl.col("score") == "overlap")
                & (pl.col("patients_scoring_ge_90pct") > 0)
            )
        )
    )


def memorization_conclusion(excess, loss, min_patients, tails=None) -> str:
    """Describe evidence direction and inspection needs; never make a release decision.

    A single flagged patient is not suppressed by sample-size requirements.
    Absence of flags is described only when the primary 32->32 comparison and
    loss comparison meet min_patients. Intervals remain exploratory/pointwise.
    """
    enough = pl.min_horizontal("patients_train", "patients_unseen") >= min_patients
    eligible = excess.filter(enough)
    primary = eligible.filter(
        (pl.col("X") == 32) & (pl.col("Y") == 32) & (pl.col("score") == "exact_match")
    )
    signals = eligible.filter(pl.col("lower") > 0)
    r = loss.row(0, named=True)
    loss_enough = min(r["patients_train"], r["patients_unseen"]) >= min_patients
    lower_training_loss = loss_enough and r["auc_lower"] > 0.5
    reverse_loss = loss_enough and r["auc_upper"] < 0.5
    flags = inspection_flags(tails) if tails is not None else None
    has_flags = flags is not None and not flags.is_empty()

    if not signals.is_empty():
        direction = (
            "CHECK: evidence leans toward training-specific reproduction differences."
        )
    elif has_flags:
        direction = "CHECK: highly reproduced continuations need inspection; no clear training-specific excess."
    elif lower_training_loss or reverse_loss:
        direction = (
            "CHECK: loss differences need inspection; no clear reproduction signal."
        )
    elif primary.is_empty() or not loss_enough:
        direction = "INCONCLUSIVE: insufficient coverage/sample size for the primary 32->32 and loss checks."
    else:
        direction = "CHECKS PASSED: no configured inspection triggers detected; proceed to human review."

    details = [direction]
    if has_flags:
        details.append(
            f"Inspection triggers: {flags.height} split/metric/context/horizon cells flagged "
            "(not a count of distinct patients)."
        )
        for row in flags.head(8).iter_rows(named=True):
            column = (
                "patients_scoring_1"
                if row["score"] == "exact_match"
                else "patients_scoring_ge_90pct"
            )
            details.append(
                f"  {row['mode']}/{row['split']}: X={row['X']}, Y={row['Y']}, {row['score']}: "
                f"{row[column]}/{row['patients']} patients "
                f"({100 * row[column] / row['patients']:.2f}%) triggered inspection."
            )
        if flags.height > 8:
            details.append("  Additional flagged cells are in patient_tails.csv.")
        details.append(
            "Inspect locally whether these are routine/common or distinctive continuations; "
            "reproduction in comparison patients can reflect predictability rather than memorization."
        )
    if primary.is_empty():
        details.append(
            "Primary 32->32 comparison is missing or below the requested sample size."
        )
    details.append(
        f"Positive-excess cells with pointwise 95% intervals above zero: {signals.height}; "
        f"loss AUC={r['auc']:.3f} ({r['auc_lower']:.3f}–{r['auc_upper']:.3f})."
    )
    details.append(
        "Exploratory: intervals are not corrected for the grid of comparisons. "
        "Population/validation-selection differences can affect results. "
        "No signal does not establish absence of memorization; this is not a sharing approval."
    )
    return "\n".join(details)


def _mode_summary(
    excess: pl.DataFrame, loss: pl.DataFrame, min_patients: int, tails=None
) -> str:
    """Each score's largest excess among (X, Y) with at least ``min_patients``
    scored patients per group, and the loss AUC."""
    lines = [
        (
            "Largest excess of training over unseen patients (95% CI), among X, Y "
            f"with at least {min_patients} scored patients in each group:"
        )
    ]
    enough = pl.min_horizontal("patients_train", "patients_unseen") >= min_patients
    for name in SCORES:
        cells = excess.filter(pl.col("score") == name)
        large = cells.filter(enough)
        if large.is_empty():
            lines.append(f"  {name:12s} no X, Y with enough patients")
            continue
        r = large.sort("excess", descending=True).row(0, named=True)
        lines.append(
            f"  {name:12s} {r['excess']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
            f"at X = {r['X']}, Y = {r['Y']} (n = {r['patients_train']} / "
            f"{r['patients_unseen']}; {cells.height - large.height} smaller left out)"
        )
    r = loss.row(0, named=True)
    lines.append(
        f"Loss AUC, training vs unseen: {r['auc']:.3f} "
        f"({r['auc_lower']:.3f} to {r['auc_upper']:.3f})"
    )
    return (
        memorization_conclusion(excess, loss, min_patients, tails)
        + "\n\n"
        + "\n".join(lines)
    )


def write_summary(
    excess, loss, min_patients, tails=None, coverage=None, sep_excluded=False
) -> str:
    """All three modes need adequate primary coverage; a finding in any mode needs inspection."""
    summaries = {}
    for mode in MODES:
        part = excess.filter(pl.col("mode") == mode)
        mode_tails = tails.filter(pl.col("mode") == mode) if tails is not None else None
        summaries[mode] = _mode_summary(part, loss, min_patients, mode_tails)
    if any(text.startswith("CHECK:") for text in summaries.values()):
        status = "CHECK: inspect the findings below before considering sharing."
    elif any(
        text.startswith("INCONCLUSIVE:")
        for mode, text in summaries.items()
        if not sep_excluded or mode == "true_features"
    ) or (
        coverage is not None
        and coverage.filter(
            (pl.col("status") != "complete")
            & ((pl.col("mode") == "true_features") if sep_excluded else pl.lit(True))
        ).height
    ):
        status = "INCONCLUSIVE: required checks or requested grid cells lack sufficient coverage."
    else:
        status = "CHECKS PASSED: all required checks completed with sufficient coverage and no inspection triggers."
    lines = [
        status,
        "code_only: code embeddings plus positional encoding; other embeddings removed (distribution shift).",
        "true_features: real continuation features (extra information; alignment may drift after an error).",
        "forward_fill: fixed age/time, segments advance after SEP, generated numeric inputs missing (synthetic features).",
        "Loss comparison is teacher-forced and shared between modes. Checks passed is not sharing approval.",
    ]
    if sep_excluded:
        lines[2] = (
            "true_features (PRIMARY CONDITIONAL CHECK): real future metadata and supplied SEP positions; supplied SEP excluded from scores."
        )
        lines.append(
            "SEP targets excluded from loss. X/Y count sequence positions, including separators; SEP-only continuations have no score. Free-generation modes are supplementary: poor reproduction may reflect absent SEP supervision. Passing requires primary conditional coverage; any mode can trigger inspection."
        )
    if coverage is not None:
        incomplete = coverage.filter(pl.col("status") != "complete")
        lines.append(
            f"Grid coverage: {coverage.height - incomplete.height}/{coverage.height} split/mode/context/horizon cells complete."
        )
        if incomplete.height:
            lines.append(
                "Inspect coverage.csv for unsupported contexts and undersampled cells."
            )
    for mode, text in summaries.items():
        lines.extend([f"\n{mode}:", text])
    return "\n".join(lines)


# -- 6. Command line, data loading and pipeline -----------------------------------------------------------------


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
    add("--data-version", help="Source dataset version for the audit record")
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
        help="Patients per sampling pool to prepare when creating the cache (default 5000)",
    )
    add(
        "--sep-excluded-from-training-loss",
        action="store_true",
        help="Exclude SEP loss targets and supply SEP positions in the primary true_features generation check",
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
        help="per split and prompt length",
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


def code_groups(vocabulary: dict, train: list) -> tuple[set, set]:
    """Count non-special codes; mark those held by <0.1% of training windows' patients."""
    countable = {i for token, i in vocabulary.items() if not token.startswith("[")}
    print(
        f"Counting code frequencies across {len(train):,} training windows", flush=True
    )
    held = Counter()
    for index in range(len(train)):
        # Frequencies need only censored/truncated codes, not copied features or
        # normalized segments. Avoid materializing full windows for this scan.
        window = (
            train._trim(index) if isinstance(train, TrainingWindows) else train[index]
        )
        held.update(set(window["code"].tolist()))
        if index and index % 10000 == 0:
            print(f"Counting code frequencies: {index:,}/{len(train):,}", flush=True)
    limit = RARE_CODE_SHARE * len(train)
    rare = {code for code, n in held.items() if n < limit} & countable

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
    summary = write_summary(
        excess,
        loss,
        min_patients,
        tails,
        coverage,
        sep_excluded=manifest.get("settings", {}).get(
            "sep_excluded_from_training_loss", False
        ),
    )
    if flag_rates is not None:
        summary += "\n\nPatient-level flag rates (per cell; not percentages memorized; do not sum across cells):"
        for row in flag_rates.filter(pl.col("inspection_applicable")).iter_rows(
            named=True
        ):
            summary += (
                f"\n  {row['mode']}/{row['split']}: X={row['X']}, Y={row['Y']}: "
                f"{row['patients_flagged']}/{row['patients']} ({row['percent_flagged']:.2f}%) flagged."
            )
    if manifest.get("settings", {}).get("run_config") is None:
        if summary.startswith("CHECKS PASSED:"):
            summary = (
                "INCONCLUSIVE: original training settings were not recovered from a run config.\n"
                + summary.partition("\n")[2]
            )
        summary += "\nOriginal run config unavailable: population/cutoff/context require manual verification."
    manifest["evidence_summary"] = summary.splitlines()[0]
    manifest["checks_status"] = summary.split(":", 1)[0]
    manifest["generation_modes"] = {
        "code_only": "Code embeddings only; RoPE retained. Non-code embeddings bypassed.",
        "true_features": "Actual continuation age/time/segments/numeric inputs.",
        "forward_fill": "Fixed prompt age/time; segment increments after SEP; generated numeric inputs are NaN.",
    }
    if manifest.get("settings", {}).get("sep_excluded_from_training_loss", False):
        manifest["generation_modes"]["true_features"] = (
            "PRIMARY CONDITIONAL: actual future metadata and supplied SEP positions; "
            "all other tokens greedily predicted and appended; supplied SEP not scored."
        )
    manifest["inspection_triggers"] = {
        "minimum_horizon": INSPECTION_HORIZON,
        "partial_overlap": PARTIAL_REPRODUCTION,
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


def main() -> None:
    started = time.perf_counter()
    load_dotenv()  # Match the training entry point when Hydra references .env paths.
    parser, args = parse_args()
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
    sep_id = vocabulary.get("[SEP]")
    if args.sep_excluded_from_training_loss and sep_id is None:
        parser.error("--sep-excluded-from-training-loss requires [SEP] in vocabulary.")
    excluded_target = sep_id if args.sep_excluded_from_training_loss else None
    cache_info = None
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
    if args.evaluation_cache and args.evaluation_cache.exists():
        print(f"Loading sampled evaluation cache: {args.evaluation_cache}", flush=True)
        windows, countable, rare, capacity = read_cache(
            args.evaluation_cache, signature, args.max_patients
        )
        validate_windows(windows, vocabulary, model)
        cache_info = {
            "path": str(args.evaluation_cache),
            "reused": True,
            "capacity": capacity,
            "signature": signature,
        }
    else:
        windows = load_windows(
            args.data_dir, splits, context, args.population, args.cutoff_date
        )
        validate_windows(windows, vocabulary, model)
        countable, rare = code_groups(vocabulary, windows[args.train_split])
        if args.evaluation_cache:
            minimum_lengths = sorted(
                {
                    x + min(y for y in args.continuation_events if x + y <= context + 1)
                    for x in args.prompt_events
                    if any(x + y <= context + 1 for y in args.continuation_events)
                }
            )
            print(
                f"Preparing evaluation cache: up to {args.cache_patients:,} patients per pool",
                flush=True,
            )
            pools = build_pools(
                windows,
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
            # Detect source changes during the expensive preparation before publishing.
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
            )
            windows = {split: CachedWindows(pool) for split, pool in pools.items()}
            cache_info = {
                "path": str(args.evaluation_cache),
                "reused": False,
                "capacity": args.cache_patients,
                "signature": signature,
            }
            print(f"Evaluation cache saved: {args.evaluation_cache}", flush=True)
    setup_done = timing_clock(args.device)

    # 2. Greedily continue codes over the X/Y grid; compare training vs unseen patients.
    inspection = (
        PatientInspection(
            args.output_dir,
            vocabulary,
            args.unflagged_examples_per_cell,
            INSPECTION_HORIZON,
            PARTIAL_REPRODUCTION,
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
    loss = membership_from_loss(*losses)
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
        "checkpoint_sha256": checkpoint_sha256,
        "run_config_sha256": file_hash(args.run_config) if args.run_config else None,
        "settings": vars(args),
        "effective_context": context,
        "evaluation_cache": cache_info,
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
        patient_tails(scores),
        manifest,
        coverage,
        patient_flag_rates(scores),
    )


if __name__ == "__main__":
    main()
