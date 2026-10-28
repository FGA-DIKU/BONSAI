"""Memorization check: does a pretrained BONSAI model reproduce its training patients?

The model gets the first X events of a patient's record and continues it for Y
events, always taking its most likely next code. Its continuation is compared
with the real record:

    exact_match  all Y events reproduced, in order
    overlap      share of the next Y real events also written, in any order
    rare_codes   share of the patient's rare codes (held by < 0.1% of training
                 patients) also written

Records are partly predictable, so each score is compared with patients the
model never trained on: memorization is the excess of the training patients.
The next-event loss of the two groups is compared too: an AUC of 0.5 means a
patient's loss says nothing about whether the model trained on them.

Records are the windows pretraining trained on. Only aggregates are written:
``excess.csv``, ``loss_check.csv`` and ``summary.txt``. The summary starts with
a short exploratory conclusion about memorization and uses only
(X, Y) with at least ``--min-patients`` scored patients per group, since long
prompts leave few patients and noisy differences.

    python -m bonsai.run.memorization_check --checkpoint best.ckpt \\
        --data-dir <dir with subject_data_{train,tuning}.pt and vocabulary.pt> \\
        --output-dir memorization
"""

import argparse
import inspect
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from bonsai.functional.subject_data import filter_subject_data
from bonsai.modules.datasets.PretrainDataset import PretrainDataset
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiPretrain,
    BonsaiValuePretrain,
)

FIELDS = ("code", "age", "abspos", "segment", "numeric_value")
SCORES = ("exact_match", "overlap", "rare_codes")
RARE_CODE_SHARE = 0.001
PROMPTS = (16, 64, 256, 1024, 4096)  # X
CONTINUATIONS = (1, 2, 4, 8, 16, 32, 64)  # Y
LOSS_COLUMNS = (
    "patients_train patients_unseen loss_train loss_unseen auc auc_lower auc_upper"
).split()
EXCESS_COLUMNS = (
    "score X Y patients_train patients_unseen train unseen excess lower upper"
).split()


# -- Model and data ---------------------------------------------------------------


def load_model(checkpoint: Path, device: str) -> BonsaiBase:
    """The pretrained model, with its next-event head, from a Lightning checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = saved["hyper_parameters"]
    if not hparams.get("causal"):
        raise ValueError("The check needs a causally pretrained model (causal: true).")
    value_model = hparams.get("value_embedding_mode")
    model_class = BonsaiValuePretrain if value_model else BonsaiPretrain
    accepted = inspect.signature(model_class).parameters
    settings = {k: v for k, v in hparams.items() if k in accepted}
    model = model_class(**settings, attention_dropout=0.0)
    weights = {
        key.removeprefix("model."): value
        for key, value in saved["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)


def training_windows(subjects, context, background_length, cutoff=None) -> list:
    """Each patient's record as pretraining saw it: ``context`` inputs plus 1 target."""
    dataset = PretrainDataset(subjects, context + 1, background_length, cutoff)
    return [dataset[i] for i in range(len(dataset))]


def sample_patients(windows: list, min_events: int, max_patients: int) -> list:
    """Up to ``max_patients`` patients with ``min_events`` or more events (seeded)."""
    eligible = [w for w in windows if len(w["code"]) >= min_events]
    if len(eligible) <= max_patients:
        return eligible
    chosen = np.random.default_rng(0).choice(len(eligible), max_patients, replace=False)
    return [eligible[i] for i in sorted(chosen)]


def make_batch(windows: list, length: int, device) -> dict:
    """The first ``length`` events of each patient, right-padded."""
    batch = {}
    for field in FIELDS:
        if all(field in w for w in windows):
            rows = [w[field][:length] for w in windows]
            batch[field] = pad_sequence(rows, batch_first=True).to(device)
    lengths = torch.tensor([min(len(w["code"]), length) for w in windows])
    positions = torch.arange(batch["code"].shape[1])
    batch["attention_mask"] = (positions < lengths[:, None]).to(device)
    return batch


# -- Generation and loss ----------------------------------------------------------


def hidden_states(model: BonsaiBase, batch: dict) -> torch.Tensor:
    """Causal attention: position i has seen events 0..i and predicts event i + 1."""
    with torch.autocast("cuda", torch.float16, enabled=batch["code"].is_cuda):
        return BonsaiBase.forward(model, batch).float()


@torch.inference_mode()
def generate(model, windows: list, prompt: int, steps: int, batch_size: int = 8):
    """[patients, steps] codes written greedily after ``prompt`` events.

    Each written code replaces the real one before the next step; times, ages
    and values stay those of the real record. These are conditioning inputs,
    not predictions; the scores assess code reproduction only.
    """
    device, written = next(model.parameters()).device, []
    for start in range(0, len(windows), batch_size):
        batch = make_batch(windows[start : start + batch_size], prompt + steps, device)
        codes = batch["code"].clone()
        for i in range(prompt, prompt + steps):
            seen = {field: values[:, :i] for field, values in batch.items()}
            seen["code"] = codes[:, :i]
            last = hidden_states(model, seen)[:, -1]
            codes[:, i] = model.pretrain_head(last).argmax(-1)
        written.append(codes[:, prompt:].cpu().numpy())
    return np.concatenate(written)


@torch.inference_mode()
def patient_losses(model, windows: list, batch_size: int = 8) -> np.ndarray:
    """Each patient's mean next-event loss over their window, as in pretraining."""
    device, losses = next(model.parameters()).device, []
    for start in range(0, len(windows), batch_size):
        part = windows[start : start + batch_size]
        batch = make_batch(part, max(len(w["code"]) for w in part) - 1, device)
        targets = [w["code"][1:] for w in part]
        targets = pad_sequence(targets, batch_first=True).to(device)
        logits = model.pretrain_head(hidden_states(model, batch)).transpose(1, 2)
        loss = F.cross_entropy(logits, targets, ignore_index=0, reduction="none")
        losses += (loss.sum(1) / (targets != 0).sum(1)).tolist()
    return np.array(losses)


# -- Scores -----------------------------------------------------------------------


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
        "exact_match": float((generated == truth).all()),
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
) -> pl.DataFrame:
    """Scores per patient, split, X and Y (kept in memory, never written)."""
    longest, rows = max(continuations), []
    for split, windows in windows_by_split.items():
        for x in prompts:
            # The same patients for every Y. Decoding is greedy, so the first Y
            # events of the longest continuation are the Y-event continuation.
            patients = sample_patients(windows, x + longest, max_patients)
            if not patients:
                continue
            generated = generate(model, patients, x, longest, batch_size)
            for written, w in zip(generated, patients):
                truth = w["code"][x : x + longest].numpy()
                for y in continuations:
                    scores = score_patient(written[:y], truth[:y], countable, rare)
                    rows.append({"split": split, "X": x, "Y": y, **scores})
    return pl.DataFrame(rows)


# -- Statistics -------------------------------------------------------------------


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
    for (x, y), part in scores.group_by("X", "Y"):
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
            rows.append((name, x, y, *counts, *excess))
    excess = pl.DataFrame(rows, schema=EXCESS_COLUMNS, orient="row")
    return excess.sort("score", "X", "Y")


def membership_from_loss(train: np.ndarray, unseen: np.ndarray) -> pl.DataFrame:
    """AUC for telling training from unseen patients by their loss (0.5 = not at
    all), with the Hanley and McNeil (1982) 95% interval."""
    n1, n2 = len(train), len(unseen)
    auc = roc_auc_score([1] * n1 + [0] * n2, -np.concatenate([train, unseen]))
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)
    half = 1.96 * math.sqrt(var / (n1 * n2))
    row = (n1, n2, train.mean(), unseen.mean(), auc, auc - half, auc + half)
    return pl.DataFrame([row], schema=LOSS_COLUMNS, orient="row")


def memorization_conclusion(
    excess: pl.DataFrame, loss: pl.DataFrame, min_patients: int
) -> str:
    """Exploratory interpretation of reproduction excess and loss membership signal.

    Intervals are pointwise across many X/Y comparisons, so this is a screening
    conclusion, not a calibrated probability or a formal test of memorization.
    """
    enough = pl.min_horizontal("patients_train", "patients_unseen") >= min_patients
    eligible = excess.filter(enough & pl.col("score").is_in(SCORES))
    r = loss.row(0, named=True)
    loss_enough = min(r["patients_train"], r["patients_unseen"]) >= min_patients
    if eligible.is_empty() or not loss_enough:
        return "Conclusion: inconclusive; too few scored patients for both checks."

    signals = sorted(set(eligible.filter(pl.col("lower") > 0)["score"].to_list()))
    loss_signal = r["auc_lower"] > 0.5
    evidence = (
        f"{len(signals)} of {eligible['score'].n_unique()} available reproduction "
        f"metrics have a positive excess with a 95% interval above zero; "
        f"loss AUC = {r['auc']:.3f} "
        f"(95% CI {r['auc_lower']:.3f}–{r['auc_upper']:.3f})."
    )
    if signals and loss_signal:
        verdict = "memorization appears likely under this check"
    elif signals:
        verdict = "possible memorization; reproduction and loss evidence are mixed"
    elif loss_signal:
        verdict = "training membership is detectable, but code memorization is unclear"
    else:
        verdict = "no clear evidence of memorization in this check"
    return (
        f"Conclusion: {verdict}. {evidence} "
        "Exploratory: intervals are not adjusted for multiple comparisons, and "
        "train/heldout differences can also reflect population differences. "
        "A negative result does not establish that memorization is unlikely."
    )


def write_summary(excess: pl.DataFrame, loss: pl.DataFrame, min_patients: int) -> str:
    """Each score's largest excess among (X, Y) with at least ``min_patients``
    scored patients per group, and the loss AUC."""
    lines = [
        "Largest excess of training over unseen patients (95% CI), among X, Y "
        f"with at least {min_patients} scored patients in each group:"
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
    return memorization_conclusion(excess, loss, min_patients) + "\n\n" + "\n".join(lines)


# -- Command line -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--checkpoint", type=Path, required=True)
    add("--data-dir", type=Path, required=True)
    add("--output-dir", type=Path, required=True)
    add("--train-split", default="train")
    add("--heldout-split", default="tuning")
    add("--population", type=Path, help="population csv used in pretraining, if any")
    add("--cutoff-date", type=date.fromisoformat, help="training.cutoff_date, if set")
    add("--prompt-events", type=int, nargs="+", default=PROMPTS)
    add("--continuation-events", type=int, nargs="+", default=CONTINUATIONS)
    add("--max-patients", type=int, default=2000, help="per split and prompt length")
    add("--min-patients", type=int, default=50, help="per group, for summary.txt")
    add("--batch-size", type=int, default=8, help="lower it if out of GPU memory")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    context = model.hparams["max_seqlen"]
    if max(args.prompt_events) + max(args.continuation_events) > context + 1:
        parser.error(f"prompt + continuation exceeds the {context + 1}-event window.")

    # The patients and windows exactly as pretraining saw them.
    splits = (args.train_split, args.heldout_split)
    subjects = {
        s: torch.load(args.data_dir / f"subject_data_{s}.pt", weights_only=False)
        for s in splits
    }
    if args.population:
        ids = pl.read_csv(args.population)["subject_id"].to_list()
        subjects = {s: filter_subject_data(v, ids) for s, v in subjects.items()}
    background = (subjects[args.train_split][0]["segment"] == 1).sum()
    day = args.cutoff_date
    cutoff = {"year": day.year, "month": day.month, "day": day.day} if day else None
    windows = {
        s: training_windows(v, context, background, cutoff) for s, v in subjects.items()
    }

    # Codes that count: no special tokens. Rare: held by < 0.1% of training patients.
    vocabulary = torch.load(args.data_dir / "vocabulary.pt", weights_only=False)
    countable = {i for token, i in vocabulary.items() if not token.startswith("[")}
    train = windows[args.train_split]
    held = Counter(code for w in train for code in set(w["code"].tolist()))
    limit = RARE_CODE_SHARE * len(train)
    rare = {code for code, n in held.items() if n < limit} & countable

    scores = score_all_patients(
        model,
        windows,
        args.prompt_events,
        sorted(args.continuation_events),
        args.max_patients,
        countable,
        rare,
        args.batch_size,
    )
    excess = excess_over_unseen(scores, *splits)
    patients = [sample_patients(windows[s], 2, args.max_patients) for s in splits]
    losses = [patient_losses(model, p, args.batch_size) for p in patients]
    loss = membership_from_loss(*losses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    excess.write_csv(args.output_dir / "excess.csv")
    loss.write_csv(args.output_dir / "loss_check.csv")
    summary = write_summary(excess, loss, args.min_patients)
    (args.output_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
