"""Audit a group-level excess cell, including patients without individual flags.

Select high-score and near-median examples from BOTH splits, then replay only
their original inference batches from the evaluation cache. No population load
or full-grid inference. Keep all outputs on the patient-data server.

python -m bonsai.run.audit_memorization_cell --report-dir RESULTS \
    --mode code_only --context 32 --horizon 16 --metric overlap --output-dir AUDIT

For every rare-code reproduction case in a cell, add --metric rare_codes
--all-rare-matches. This ignores the top-score cap and minimum individual-flag
horizon, selecting all saved positive rare-code scores plus zero-score controls.
Counts and decoded rare-code evidence go in audit.txt and rare_code_matches.csv;
rare_code_review_cases.csv lists every selected positive case. Old caches retain
rare membership but not exact population holder counts; unavailable counts are
explicit. New evaluation caches preserve these counts during existing preparation.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

from bonsai.functional.memorization_cache import read_cache
from bonsai.functional.memorization_inspection import PatientInspection
from bonsai.run.inspect_memorization import write_audit
from bonsai.run.memorization_check import (
    INSPECTION_HORIZON,
    MODES,
    PARTIAL_REPRODUCTION,
    SCORES,
    difference_in_means,
    file_hash,
    generate,
    load_model,
    sample_patients,
    score_patient,
    validate_windows,
)


def select_examples(
    frame, splits, metric, top, median, seed, subject_ids=None, all_rare_matches=False
):
    """Rank all scored patients; never restrict selection to individual flags."""
    selected, statistics = [], {}
    for offset, split in enumerate(splits):
        part = frame.filter(pl.col("split") == split).sort("subject_id")
        if part.is_empty():
            raise ValueError(f"No finite {metric} scores for {split} in this cell.")
        if part["subject_id"].n_unique() != part.height:
            raise ValueError(
                "Duplicate patient/cell scores; cannot identify a unique original trajectory."
            )
        values = part[metric].to_numpy()
        centre = float(np.median(values))
        statistics[split] = {
            "patients": len(values),
            "mean": float(values.mean()),
            "median": centre,
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "positive_patients": int((values > 0).sum()),
        }
        records = part.to_dicts()
        rng = np.random.default_rng(seed + offset)
        rng.shuffle(records)  # Uniform, reproducible tie-breaking.
        if all_rare_matches:
            ranked = sorted(records, key=lambda row: -row[metric])
            choices = [
                (row, "rare_code_match_review") for row in ranked if row[metric] > 0
            ]
            controls = [row for row in records if row[metric] == 0]
            choices += [(row, "zero_score_control") for row in controls[:median]]
        elif subject_ids:
            choices = [
                (row, "requested_id")
                for row in records
                if row["subject_id"] in subject_ids
            ]
        else:
            ranked = sorted(records, key=lambda row: -row[metric])
            choices = [(row, "top_score") for row in ranked[:top]]
            chosen = {row["subject_id"] for row, _ in choices}
            controls = sorted(
                (row for row in records if row["subject_id"] not in chosen),
                key=lambda row: abs(row[metric] - centre),
            )
            choices += [(row, "near_median") for row in controls[:median]]
        for row, reason in choices:
            selected.append(
                {
                    **row,
                    "selection_reason": reason,
                    "signed_mean_contribution": (1 if offset == 0 else -1)
                    * row[metric]
                    / len(values),
                }
            )
    if subject_ids:
        missing = subject_ids - {row["subject_id"] for row in selected}
        if missing:
            raise ValueError(
                f"Requested IDs have no finite score in this cell: {sorted(missing)}"
            )
    return selected, statistics


def rare_code_evidence(truth, predicted, rare, vocabulary, metadata=None, decoded=None):
    """Unordered occurrence matches; a generated occurrence can match only once."""
    true_counts = Counter(int(code) for code in truth if int(code) in rare)
    predicted_counts = Counter(int(code) for code in predicted if int(code) in rare)
    tokens = (
        decoded
        if decoded is not None
        else {identifier: token for token, identifier in vocabulary.items()}
    )
    population = metadata.get("training_patients") if metadata else None
    frequencies = metadata.get("patient_counts", {}) if metadata else {}
    codes = []
    for code in sorted(true_counts.keys() | predicted_counts.keys()):
        count = frequencies.get(code)
        codes.append(
            {
                "code_id": code,
                "token": tokens[code],
                "true_occurrences": true_counts[code],
                "predicted_occurrences": predicted_counts[code],
                "matched_occurrences": min(true_counts[code], predicted_counts[code]),
                "training_patient_count": count,
                "training_patients": population,
                "training_patient_fraction": count / population
                if count is not None and population
                else None,
            }
        )
    return {
        "true_rare_occurrences": sum(true_counts.values()),
        "predicted_rare_occurrences": sum(predicted_counts.values()),
        "matched_rare_occurrences": sum(row["matched_occurrences"] for row in codes),
        "population_frequencies_available": metadata is not None,
        "codes": codes,
    }


def original_batches(dataset, settings, prompt, horizon, context):
    """Reconstruct original grouping/order/batch shape to minimize replay drift."""
    horizons = sorted(
        y for y in settings["continuation_events"] if prompt + y <= context + 1
    )
    if horizon not in horizons:
        raise ValueError("Requested horizon was not evaluated in the original grid.")
    records = sample_patients(dataset, prompt + min(horizons), settings["max_patients"])
    groups = defaultdict(list)
    for row in records:
        longest = max(y for y in horizons if prompt + y <= len(row["code"]))
        groups[longest].append(row)
    batch = settings["batch_size"]
    for longest in sorted(groups):
        if longest < horizon:
            continue
        rows = groups[longest]
        for start in range(0, len(rows), batch):
            yield rows[start : start + batch]


def _same(left, right):
    if left is None or not math.isfinite(float(left)):
        return right is None or not math.isfinite(float(right))
    return right is not None and math.isclose(
        float(left), float(right), rel_tol=1e-8, abs_tol=1e-8
    )


def audit_cell(args):
    report = args.report_dir
    manifest = json.loads((report / "run_manifest.json").read_text())
    settings = manifest["settings"]
    if manifest.get("analysis_role") == "primary_patient_code_only":
        if args.context != 0 or args.mode != "code_only":
            raise ValueError("This primary report contains code_only X=0 cells.")
        frame = pl.read_csv(
            report / "patient_scores.csv",
            schema_overrides={
                "subject_id": pl.String,
                **{name: pl.Float64 for name in SCORES},
            },
            null_values=["nan", "NaN"],
        ).filter((pl.col("Y") == args.horizon) & pl.col(args.metric).is_finite())
        cohort = args.cohort or manifest.get("analysis_cohort", "combined")
        if cohort != "combined":
            frame = frame.filter(pl.col("sampling_cohort") == cohort)
        selected, _ = select_examples(
            frame,
            (settings["train_split"], settings["heldout_split"]),
            args.metric,
            args.top_per_split,
            args.median_per_split,
            args.seed,
            set(args.subject_id) if args.subject_id else None,
            args.all_rare_matches,
        )
        from bonsai.run.zero_context_check import audit_stored

        audit_stored(
            report,
            args.output_dir,
            selected=[{**row, "audit_metric": args.metric} for row in selected],
        )
        frame.write_csv(args.output_dir / "cell_patient_scores.csv")
        if args.all_rare_matches:
            frame.filter(pl.col("rare_codes") > 0).write_csv(
                args.output_dir / "rare_code_review_cases.csv"
            )
        arrays = [
            frame.filter(pl.col("split") == split)[args.metric].to_numpy()
            for split in (settings["train_split"], settings["heldout_split"])
        ]
        estimate = (
            difference_in_means(*arrays, args.metric == "exact_match")
            if min(map(len, arrays)) >= 2
            else (math.nan, math.nan, math.nan)
        )
        (args.output_dir / "summary.txt").write_text(
            f"CONTEXT-ZERO CELL: cohort={cohort}, Y={args.horizon}, metric={args.metric}\nExcess={estimate[0]:+.6f} [{estimate[1]:+.6f}, {estimate[2]:+.6f}]; contributing patients={len(arrays[0])}/{len(arrays[1])}.\nStored original assigned samples; no replay. All contributing IDs: cell_patient_scores.csv; selected cases: audit.txt.\n"
        )
        return
    splits = (settings["train_split"], settings["heldout_split"])
    cohort = args.cohort or manifest.get("analysis_cohort", "combined")
    frame = pl.read_csv(
        report / "patient_scores.csv",
        schema_overrides={
            "subject_id": pl.String,
            **{name: pl.Float64 for name in SCORES},
        },
        null_values=["nan", "NaN"],
    ).filter(
        (pl.col("mode") == args.mode)
        & (pl.col("X") == args.context)
        & (pl.col("Y") == args.horizon)
        & pl.col(args.metric).is_finite()
    )
    if cohort != "combined":
        frame = frame.filter(pl.col("sampling_cohort") == cohort)
    selected, statistics = select_examples(
        frame,
        splits,
        args.metric,
        args.top_per_split,
        args.median_per_split,
        args.seed,
        set(args.subject_id) if args.subject_id else None,
        args.all_rare_matches,
    )
    cache_info = manifest.get("evaluation_cache") or {}
    cache_path = args.evaluation_cache or (
        Path(cache_info["path"]) if cache_info.get("path") else None
    )
    if cache_path is None or cache_info.get("signature") is None:
        raise ValueError(
            "This report has no verified evaluation-cache provenance. A cached original run is required."
        )
    checkpoint = args.checkpoint or Path(manifest["checkpoint"])
    vocabulary_path = args.vocabulary or Path(settings["data_dir"]) / "vocabulary.pt"
    if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError(
            "Checkpoint differs from the original run; cannot audit its original results."
        )
    if file_hash(vocabulary_path) != manifest["vocabulary_sha256"]:
        raise ValueError("Vocabulary differs from the original run.")
    datasets, countable, rare, _ = read_cache(
        cache_path, cache_info["signature"], settings["max_patients"]
    )
    vocabulary = torch.load(vocabulary_path, map_location="cpu", weights_only=False)
    decoded = {identifier: token for token, identifier in vocabulary.items()}
    device = args.device or settings["device"]
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA unavailable; use --device cpu and check replay-agreement labels."
            )
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError(
                    "Original CUDA inference requires native BF16 support."
                )
    model = load_model(checkpoint, device)
    sep_id = vocabulary.get("[SEP]")
    supplied = args.mode == "true_features" and settings.get(
        "sep_excluded_from_training_loss", False
    )
    pending = {(row["split"], row["subject_id"]): row for row in selected}
    if not pending:
        raise ValueError("No patients selected.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Use a fresh output directory; audit files are not overwritten."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exporter = PatientInspection(
        args.output_dir,
        vocabulary,
        len(selected),
        INSPECTION_HORIZON,
        PARTIAL_REPRODUCTION,
        settings.get("min_unique_overlap_tokens", 3),
    )
    replayed, code_rows, batches = [], [], 0
    rare_metadata = getattr(datasets[splits[0]], "rare_metadata", None)
    for split in splits:
        for batch in original_batches(
            datasets[split],
            settings,
            args.context,
            args.horizon,
            manifest["effective_context"],
        ):
            if not any((split, str(row["subject_id"])) in pending for row in batch):
                continue
            validate_windows({split: batch}, vocabulary, model)
            print(
                f"Replaying {split}, X={args.context}, Y={args.horizon}: original batch of {len(batch)} patients",
                flush=True,
            )
            predictions = generate(
                model,
                batch,
                args.context,
                args.horizon,
                settings["batch_size"],
                mode=args.mode,
                sep_token_id=sep_id,
                supply_sep=supplied,
            )
            batches += 1
            for window, written in zip(batch, predictions):
                key = (split, str(window["subject_id"]))
                if key not in pending:
                    continue
                saved = pending.pop(key)
                truth = window["code"][
                    args.context : args.context + args.horizon
                ].numpy()
                predicted = written
                if supplied:
                    mask = truth != sep_id
                    truth, predicted = truth[mask], predicted[mask]
                scores = score_patient(predicted, truth, countable, rare)
                agrees = all(
                    _same(saved[name], scores[name])
                    for name in (
                        *SCORES,
                        "unique_predicted_tokens",
                        "unique_reference_tokens",
                    )
                    if name in saved
                )
                context = {
                    "reason": saved["selection_reason"],
                    "metric": args.metric,
                    "saved_score": saved[args.metric],
                    "replayed_score": scores[args.metric],
                    "replay_matches_saved_scores": agrees,
                    "signed_mean_contribution": saved["signed_mean_contribution"],
                }
                audit_window = {**window, "audit_selection": context}
                if args.metric == "rare_codes":
                    evidence = rare_code_evidence(
                        truth, predicted, rare, vocabulary, rare_metadata, decoded
                    )
                    audit_window["rare_code_evidence"] = evidence
                    for entry in evidence["codes"]:
                        code_rows.append(
                            {
                                "split": split,
                                "subject_id": saved["subject_id"],
                                "selection_reason": saved["selection_reason"],
                                "replay_matches_saved_scores": agrees,
                                **entry,
                            }
                        )
                    context["matched_rare_occurrences"] = evidence[
                        "matched_rare_occurrences"
                    ]
                    context["true_rare_occurrences"] = evidence["true_rare_occurrences"]
                exporter.record(
                    audit_window,
                    written,
                    split,
                    args.mode,
                    args.context,
                    args.horizon,
                    scores,
                    supplied,
                    sep_id,
                )
                replayed.append(
                    {"split": split, "subject_id": saved["subject_id"], **context}
                )
    exporter.close()
    if pending:
        raise ValueError(
            "Selected IDs could not be reconstructed from the original cache/grid. Audit is incomplete."
        )
    if file_hash(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during replay; audit is invalid.")
    train = frame.filter(pl.col("split") == splits[0])[args.metric].to_numpy()
    tuning = frame.filter(pl.col("split") == splits[1])[args.metric].to_numpy()
    estimate = (
        difference_in_means(train, tuning, args.metric == "exact_match")
        if min(len(train), len(tuning)) >= 2
        else (float(train.mean() - tuning.mean()), math.nan, math.nan)
    )
    mismatches = sum(not row["replay_matches_saved_scores"] for row in replayed)
    frame.write_csv(args.output_dir / "cell_patient_scores.csv")
    if args.all_rare_matches:
        frame.filter(pl.col("rare_codes") > 0).write_csv(
            args.output_dir / "rare_code_review_cases.csv"
        )
    pl.DataFrame(replayed).write_csv(args.output_dir / "selected_patients.csv")
    if code_rows:
        pl.DataFrame(code_rows).write_csv(args.output_dir / "rare_code_matches.csv")
    information = {
        "mode": args.mode,
        "X": args.context,
        "Y": args.horizon,
        "metric": args.metric,
        "cohort": cohort,
        "statistics": statistics,
        "excess": estimate[0],
        "lower": estimate[1],
        "upper": estimate[2],
        "selected_examples": len(replayed),
        "original_batches_replayed": batches,
        "replay_mismatches": mismatches,
        "selection": "all patients with a saved positive rare-code score, plus zero-score controls, in both splits"
        if args.all_rare_matches
        else "top scores and near-median controls, both splits; examples are explanatory, not a representative sample",
        "all_rare_matches": args.all_rare_matches,
        "population_frequencies_available": rare_metadata is not None,
        "source_report": str(report.resolve()),
        "source_diversity_filter": manifest.get("diversity_filter"),
        "device": device,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "cell_audit": information,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "inspection_triggers": manifest.get("inspection_triggers", {}),
            },
            indent=2,
        )
        + "\n"
    )
    text = [
        f"CELL AUDIT: {args.mode} X={args.context} Y={args.horizon} metric={args.metric} cohort={cohort}",
        f"Saved-score excess: {estimate[0]:+.6f}; pointwise 95% interval [{estimate[1]:+.6f}, {estimate[2]:+.6f}].",
    ]
    for split, values in statistics.items():
        text.append(
            f"{split}: n={values['patients']}, mean={values['mean']:.6f}, median={values['median']:.6f}, p05={values['p05']:.6f}, p95={values['p95']:.6f}, max={values['maximum']:.6f}; positive scores={values['positive_patients']}/{values['patients']} ({100 * values['positive_patients'] / values['patients']:.2f}%)."
        )
    text += [
        f"Selected {len(replayed)} examples; replayed {batches} original batches; score mismatches={mismatches}.",
        "Scores/distribution use ALL finite patient scores in the cell, not just examples or individual flags.",
        "Examples explain a group comparison; they are not labelled memorized. Controls are selected for comparison, not as population estimates.",
        "Replay disagreement is labelled per case: original predictions were not stored for every patient. Do not treat a differing replay as the original trajectory.",
        "All shown trajectories are regenerated. Agreement of saved scores is a consistency check, not proof of token-identical original predictions.",
        "Open audit.txt in vim; cell_patient_scores.csv contains all contributing IDs/scores; selected_patients.csv identifies selected examples.",
    ]
    if args.metric == "rare_codes":
        text += [
            "Rare-code evidence: per-patient true/predicted/matched occurrence counts appear in audit.txt and rare_code_matches.csv.",
            "Population frequencies: available from cache."
            if rare_metadata is not None
            else "Exact population frequencies were not retained in this older cache. Rare-code membership is preserved; counts are unavailable, not inferred from the evaluation sample.",
        ]
    if args.all_rare_matches:
        text.append(
            "ALL positive saved rare-code cases were selected, with no top-score cap or minimum horizon. Zero-score controls are additional examples. Rare-code review is not an individual memorization verdict."
        )
    (args.output_dir / "summary.txt").write_text("\n".join(text) + "\n")
    write_audit(
        args.output_dir,
        args.output_dir / "audit.txt",
        settings.get("min_unique_overlap_tokens", 3),
    )
    print("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--metric", choices=SCORES, default="overlap")
    parser.add_argument("--cohort", choices=("combined", "random", "uncommon"))
    parser.add_argument("--top-per-split", type=int, default=5)
    parser.add_argument("--median-per-split", type=int, default=3)
    parser.add_argument("--subject-id", nargs="+")
    parser.add_argument(
        "--all-rare-matches",
        action="store_true",
        help="Audit EVERY patient with rare_codes > 0 in this cell, plus zero-score controls; requires --metric rare_codes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluation-cache", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    args = parser.parse_args()
    if args.all_rare_matches and (args.metric != "rare_codes" or args.subject_id):
        parser.error(
            "--all-rare-matches requires --metric rare_codes and cannot be combined with --subject-id."
        )
    if (
        (args.context < 0 or args.horizon < 1)
        or min(args.top_per_split, args.median_per_split) < 0
        or not (
            args.all_rare_matches
            or args.subject_id
            or args.top_per_split + args.median_per_split
        )
    ):
        parser.error(
            "Positive context/horizon and at least one selected example are required."
        )
    audit_cell(args)


if __name__ == "__main__":
    main()
