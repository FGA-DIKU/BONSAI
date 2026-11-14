"""Context-zero cell on the SAME fixed patients as the supplemental checks.

python -m bonsai.run.zero_context_check --report-dir PREPARATION_OR_SUPPLEMENTAL \
    --output-dir ZERO_CONTEXT --batch-size 128 --device cuda

One independently sampled continuation per eligible patient, assigned before
scoring. The only model input is the fixed public DOB code. Scores/excess use the shared metrics, not a search against other records. Cached draws support
--reuse-generations ZERO_CONTEXT and --audit-only ZERO_CONTEXT --subject-id ID.
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from bonsai.functional.memorization_cache import cohort_identity, load_fixed_references
from bonsai.functional.memorization_generation import sample_codes
from bonsai.functional.memorization_inspection import PatientInspection
from bonsai.functional.memorization_metrics import (
    SCORES,
    assess_comparisons,
    excess_over_unseen,
    grid_coverage,
    patient_flag_rates,
    patient_tails,
    score_patient,
)
from bonsai.run.inspect_memorization import write_audit
from bonsai.run.memorization_check import file_hash, load_model

HORIZONS = (16, 32, 64, 128, 256, 512)


def reference_continuations(records, seed_id, seed_token):
    """Locate the public anchor only in references; never feed their tokens back."""
    rows, assignments, invalid = [], [], []
    for split, patients in records.items():
        for original in patients:
            positions = np.flatnonzero(np.asarray(original["code"]) == seed_id)
            if len(positions) != 1:
                invalid.append((split, str(original["subject_id"]), len(positions)))
                continue
            offset = int(positions[0]) + 1
            index = len(rows)
            row = {
                "code": original["code"][offset:],
                "subject_id": original["subject_id"],
                "sampling_cohort": original["sampling_cohort"],
                "split": split,
                "reference_offset": offset,
                "reference_history": original["code"][:offset],
                "public_seed": {"id": seed_id, "token": seed_token},
                "generation_row": index,
            }
            rows.append(row)
            assignments.append(
                {
                    "row_index": index,
                    "split": split,
                    "subject_id": str(original["subject_id"]),
                    "sampling_cohort": original["sampling_cohort"],
                    "reference_offset": offset,
                    "reference_length": len(row["code"]),
                }
            )
    if invalid:
        raise ValueError(
            f"{len(invalid)} selected records lack exactly one {seed_token!r} anchor. No patients were replaced; check the source representation before running context zero."
        )
    return rows, assignments


def cached_draws(args, source, rows, vocabulary_hash):
    """A fixed row/ID assignment; short records decode only their longest eligible Y."""
    identity = file_hash(args.output_dir / "generation_assignment.csv")
    expected = {
        "checkpoint_sha256": source["checkpoint_sha256"],
        "vocabulary_sha256": vocabulary_hash,
        "assignment_sha256": identity,
        "start_token": args.start_token,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.reuse_generations:
        generation = json.loads(
            (args.reuse_generations / "generation_manifest.json").read_text()
        )
        if generation.get("design") != "one_draw_per_patient" or any(
            generation.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "Stored draws have different patients/assignment/model/sampling settings. An independent extraction pool cannot substitute for assigned patient draws."
            )
        if file_hash(Path(generation["path"])) != generation["sha256"]:
            raise ValueError("Stored predictions changed.")
        samples = np.load(generation["path"], mmap_mode="r")
        generated = generation["generated_positions"]
        if (
            samples.ndim != 2
            or len(samples) != len(rows)
            or len(generated) != len(rows)
        ):
            raise ValueError("Stored draw dimensions do not match patient assignments.")
        for row, available in zip(rows, generated):
            required = max(
                (h for h in args.horizons if h <= len(row["code"])), default=0
            )
            if required > available:
                raise ValueError(
                    "Some patient draws do not cover the requested horizon; new inference is required."
                )
        return samples, generation
    checkpoint = args.checkpoint or Path(source["checkpoint"])
    if file_hash(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from source report.")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable.")
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("The selected CUDA device needs native BF16 support.")
    model = load_model(checkpoint, args.device)
    if max(args.horizons) > model.hparams["max_seqlen"]:
        raise ValueError("Requested generation horizon exceeds checkpoint context.")
    generated = [
        max((h for h in args.horizons if h <= len(row["code"])), default=0)
        for row in rows
    ]
    partial = args.output_dir / "generations.partial.npy"
    samples = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.int32, shape=(len(rows), max(args.horizons))
    )
    samples[:] = -1  # Not generated/scored; never interpreted as a code.
    started = time.perf_counter()
    for horizon in sorted(set(generated) - {0}):
        indices = np.flatnonzero(np.asarray(generated) == horizon)
        print(
            f"Context zero: sampling {len(indices):,} assigned patients to Y={horizon}; no patient tokens/features supplied",
            flush=True,
        )
        temporary = args.output_dir / f"group_{horizon}.partial.npy"
        group = sample_codes(
            model,
            rows[0]["public_seed"]["id"],
            len(indices),
            horizon,
            args.batch_size,
            args.temperature,
            args.seed + horizon,
            temporary,
        )
        for start in range(0, len(indices), 4096):
            selected = indices[start : start + 4096]
            samples[selected, :horizon] = group[start : start + len(selected)]
        del group
        temporary.unlink()
    samples.flush()
    del samples, model
    path = args.output_dir / "generations.npy"
    partial.rename(path)
    if file_hash(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during generation.")
    generation = {
        **expected,
        "design": "one_draw_per_patient",
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "generated_positions": generated,
        "model_max_seqlen": source["effective_context"],
        "sampling_seconds": time.perf_counter() - started,
        "batch_size": args.batch_size,
        "device": args.device,
        "group_seed_policy": "base seed + longest eligible horizon; full softmax; no token masks/top-k",
        "patient_context": 0,
        "model_seed_length": 1,
    }
    return np.load(path, mmap_mode="r"), generation


def write_comparisons(directory, scores, manifest, minimum_diversity=None):
    directory.mkdir(parents=True, exist_ok=True)
    settings = manifest["settings"]
    splits = settings["train_split"], settings["heldout_split"]
    excess = excess_over_unseen(scores, *splits)
    coverage = grid_coverage(
        scores,
        [0],
        settings["continuation_events"],
        splits,
        manifest["effective_context"] - 1,
        settings["min_patients"],
        minimum_diversity=minimum_diversity,
        modes=("code_only",),
    )
    status, flags, assessment = assess_comparisons(
        excess,
        coverage,
        settings["min_patients"],
        complete=manifest["sampling_complete"],
    )
    manifest = json.loads(json.dumps(manifest, default=str))
    manifest.update(
        checks_status=status,
        evidence_summary=f"{status}: context-zero per-patient train–tuning excess comparisons.",
    )
    scores.write_csv(directory / "patient_scores.csv")
    excess.write_csv(directory / "excess.csv")
    coverage.write_csv(directory / "coverage.csv")
    patient_tails(scores, settings["min_unique_overlap_tokens"]).write_csv(
        directory / "patient_tails.csv"
    )
    rates = patient_flag_rates(scores, settings["min_unique_overlap_tokens"])
    rates.write_csv(directory / "patient_flag_rates.csv")
    flags.write_csv(directory / "flagged_comparisons.csv")
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        manifest["evidence_summary"],
        *assessment,
        "Same fixed patients/cohort labels as the source cache; one independently sampled continuation per eligible patient.",
        "Only the public DOB code enters the model. Patient context=0; model seed length=1; target starts immediately after that patient's DOB anchor.",
        "Exact match, occurrence-limited overlap, rare-code scores and mean-excess intervals use the existing functions. Rare-score denominators require true rare targets.",
        "Raw horizons count sequence positions, including SEP. No true features or SEP positions are supplied. The seed is not scored.",
        "Pointwise intervals screen differences, not equivalence or absence of memorization. Individual reproduction flags remain audit evidence.",
        "One random draw per target can miss a different memorized patient that was generated; long exact-match scores can have low sensitivity. This is a per-patient excess diagnostic, not a database-wide extraction search.",
        "The loss/AUC comparison belongs to the supplemental checks and is not recomputed for each context.",
    ]
    for row in flags.iter_rows(named=True):
        lines.append(
            f"CHECK X=0 Y={row['Y']} {row['score']}: excess={100 * row['excess']:+.4f} pp [{100 * row['lower']:+.4f}, {100 * row['upper']:+.4f}]; n={row['patients_train']}/{row['patients_unseen']}."
        )
    for row in rates.iter_rows(named=True):
        if row["inspection_applicable"]:
            lines.append(
                f"{row['split']} Y={row['Y']}: exact match={row['patients_exact_flagged']}/{row['patients']} ({row['percent_exact_flagged']:.3f}%); low diversity among exact={row['patients_flagged_low_diversity']}/{row['patients_exact_flagged']}; total low diversity={row['patients_low_diversity']}/{row['patients']}."
            )
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    return status


def audit_stored(report_dir, output_dir, subject_ids=None, selected=None):
    """Recover original assigned draws and references, without model inference."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a fresh audit directory.")
    metadata = json.loads((report_dir / "run_manifest.json").read_text())
    generation = metadata["generation"]
    if file_hash(Path(generation["path"])) != generation["sha256"]:
        raise ValueError("Stored draws changed.")
    vocabulary_path = Path(metadata["vocabulary"])
    if file_hash(vocabulary_path) != metadata["vocabulary_sha256"]:
        raise ValueError("Vocabulary changed.")
    vocabulary = torch.load(vocabulary_path, map_location="cpu", weights_only=False)
    records, countable, rare, _ = load_fixed_references(
        metadata, metadata["settings"]["max_patients"]
    )
    rows, _ = reference_continuations(
        records, vocabulary[generation["start_token"]], generation["start_token"]
    )
    lookup = {(row["split"], str(row["subject_id"])): row for row in rows}
    if selected is None:
        frame = pl.read_csv(
            report_dir / "patient_scores.csv",
            schema_overrides={
                "subject_id": pl.String,
                **{name: pl.Float64 for name in SCORES},
            },
            null_values=["nan", "NaN"],
        )
        if subject_ids:
            frame = frame.filter(pl.col("subject_id").is_in(subject_ids))
            if set(subject_ids) - set(frame["subject_id"].to_list()):
                raise ValueError(
                    "Some requested IDs have no evaluated cells in this report."
                )
        else:
            frame = frame.filter(pl.col("flagged"))
        selected = frame.to_dicts()
    samples = np.load(generation["path"], mmap_mode="r")
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = PatientInspection(
        output_dir,
        vocabulary,
        0,
        32,
        0.9,
        metadata["settings"]["min_unique_overlap_tokens"],
        stream_unflagged=True,
    )
    for saved in selected:
        window = dict(lookup[(saved["split"], str(saved["subject_id"]))])
        horizon = saved["Y"]
        written = samples[window["generation_row"], :horizon]
        metrics = score_patient(
            written, window["code"][:horizon].numpy(), countable, rare
        )
        for name in SCORES:
            original, actual = saved[name], metrics[name]
            same = (
                not math.isfinite(actual)
                if original is None or not math.isfinite(original)
                else math.isclose(original, actual, abs_tol=1e-8)
            )
            if not same:
                raise ValueError("Stored patient metrics disagree with reference data.")
        if reason := saved.get("selection_reason"):
            metric = saved.get(
                "audit_metric",
                "rare_codes" if reason == "rare_code_match_review" else "exact_match",
            )
            window["audit_selection"] = {
                "reason": reason,
                "metric": metric,
                "saved_score": saved[metric],
                "replayed_score": metrics[metric],
                "replay_matches_saved_scores": True,
                "stored_draw": True,
            }
        exporter.record(
            window,
            written,
            saved["split"],
            "code_only",
            0,
            horizon,
            metrics,
            False,
            vocabulary.get("[SEP]"),
        )
    exporter.close()
    write_audit(
        output_dir,
        output_dir / "audit.txt",
        metadata["settings"]["min_unique_overlap_tokens"],
    )


def run(args):
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("Use a fresh output directory.")
    if args.plots and importlib.util.find_spec("matplotlib") is None:
        raise ValueError(
            "Install '.[plots]' or omit --plots and plot saved reports later."
        )
    started = time.perf_counter()
    source = json.loads((args.report_dir / "run_manifest.json").read_text())
    if source.get("analysis_cohort") or source.get("diversity_filter"):
        raise ValueError("Use the unfiltered combined source report/preparation.")
    maximum = args.max_patients or source["settings"]["max_patients"]
    records, countable, rare, cache = load_fixed_references(
        source, maximum, args.evaluation_cache
    )
    identity = cohort_identity(records)
    vocabulary_path = (
        args.vocabulary or Path(source["settings"]["data_dir"]) / "vocabulary.pt"
    )
    vocabulary_hash = file_hash(vocabulary_path)
    if vocabulary_hash != source["vocabulary_sha256"]:
        raise ValueError("Vocabulary differs from source report.")
    vocabulary = torch.load(vocabulary_path, map_location="cpu", weights_only=False)
    if args.start_token not in vocabulary:
        raise ValueError(
            "Public start token absent; use its exact vocabulary spelling."
        )
    if max(args.horizons) > source["effective_context"]:
        raise ValueError("Requested horizon exceeds the original training context.")
    rows, assignments = reference_continuations(
        records, vocabulary[args.start_token], args.start_token
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(assignments).write_csv(args.output_dir / "generation_assignment.csv")
    samples, generation = cached_draws(args, source, rows, vocabulary_hash)
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(generation, indent=2) + "\n"
    )
    inspection = PatientInspection(
        args.output_dir,
        vocabulary,
        args.unflagged_examples,
        32,
        0.9,
        args.min_unique_tokens,
    )
    for row, available in zip(rows, generation["generated_positions"]):
        for horizon in args.horizons:
            if horizon > len(row["code"]):
                continue
            if horizon > available:
                raise ValueError(
                    "Missing stored continuation; no partial result can pass."
                )
            written = samples[row["generation_row"], :horizon]
            metrics = score_patient(
                written, row["code"][:horizon].numpy(), countable, rare
            )
            inspection.record(
                row,
                written,
                row["split"],
                "code_only",
                0,
                horizon,
                metrics,
                False,
                vocabulary.get("[SEP]"),
            )
    inspection.close()
    frame = pl.read_csv(
        args.output_dir / "patient_scores.csv",
        schema_overrides={
            "subject_id": pl.String,
            **{metric: pl.Float64 for metric in SCORES},
        },
        null_values=["nan", "NaN"],
    ).with_columns(pl.col(*SCORES).fill_null(float("nan")))
    settings = {
        **source["settings"],
        "max_patients": maximum,
        "prompt_events": [0],
        "continuation_events": args.horizons,
        "output_dir": str(args.output_dir.resolve()),
        "min_unique_overlap_tokens": args.min_unique_tokens,
    }
    manifest = {
        **source,
        "settings": settings,
        "analysis_role": "primary_patient_code_only",
        "completion_status": "scoring_complete",
        "cohort_id_sha256": identity,
        "evaluation_cache": cache,
        "generation": generation,
        "vocabulary": str(vocabulary_path.resolve()),
        "generation_assignment": str(
            (args.output_dir / "generation_assignment.csv").resolve()
        ),
        "source_report": str(args.report_dir.resolve()),
        "patient_context": 0,
        "model_seed_length": 1,
        "patient_inspection": {
            "scope": "assigned stored stochastic draws; references never fed to generation"
        },
        "sampling_complete": source.get("sampling_complete", True)
        and source["settings"].get("run_config") is not None,
        "inspection_triggers": {
            "minimum_unique_predicted_clinical_tokens_for_overlap": args.min_unique_tokens
        },
        "seconds": {"total": time.perf_counter() - started},
    }
    manifest = json.loads(json.dumps(manifest, default=str))
    for key in (
        "timing",
        "cohort_statuses",
        "analysis_cohort",
        "diversity_filter",
        "diversity_filters",
    ):
        manifest.pop(key, None)
    views = [(Path(), frame)]
    if args.min_diversity_tokens:
        views.append(
            (
                Path("diversity_filtered"),
                frame.filter(
                    pl.col(f"unique_{args.diversity_basis}_tokens")
                    >= args.min_diversity_tokens
                ),
            )
        )
    for view, data in views:
        statuses = {}
        base = args.output_dir / view
        if view != Path():
            exclusions = (
                frame.group_by("mode", "split", "X", "Y")
                .agg(
                    pl.len().alias("patients_before"),
                    (
                        pl.col(f"unique_{args.diversity_basis}_tokens")
                        >= args.min_diversity_tokens
                    )
                    .sum()
                    .alias("patients_retained"),
                )
                .with_columns(
                    (pl.col("patients_before") - pl.col("patients_retained")).alias(
                        "patients_excluded"
                    )
                )
            )
            base.mkdir(parents=True, exist_ok=True)
            exclusions.sort("mode", "split", "X", "Y").write_csv(
                base / "diversity_exclusions.csv"
            )
        metadata = json.loads(json.dumps(manifest))
        if view != Path():
            metadata["diversity_filter"] = {
                "basis": args.diversity_basis,
                "minimum_unique_tokens": args.min_diversity_tokens,
            }
        for cohort in ("combined", "random", "uncommon"):
            part = (
                data
                if cohort == "combined"
                else data.filter(pl.col("sampling_cohort") == cohort)
            )
            destination = base if cohort == "combined" else base / "cohorts" / cohort
            child = json.loads(json.dumps(metadata))
            child["reanalysis"] = {
                "source_trajectory_exports": str(args.output_dir.resolve())
            }
            if destination == args.output_dir:
                child.pop("reanalysis")
            if cohort != "combined":
                child["analysis_cohort"] = cohort
            statuses[cohort] = write_comparisons(
                destination,
                part,
                child,
                args.min_diversity_tokens if view != Path() else None,
            )
        overall = (
            "CHECK"
            if "CHECK" in statuses.values()
            else "INCONCLUSIVE"
            if "INCONCLUSIVE" in statuses.values()
            else "CHECKS PASSED"
        )
        root_manifest = json.loads((base / "run_manifest.json").read_text())
        root_manifest.update(
            checks_status=overall,
            cohort_statuses=statuses,
            evidence_summary=f"{overall}: context-zero per-patient comparisons across all three cohorts.",
        )
        (base / "run_manifest.json").write_text(
            json.dumps(root_manifest, indent=2) + "\n"
        )
        text = (
            root_manifest["evidence_summary"]
            + "\n"
            + "Cohort statuses: "
            + ", ".join(f"{name}={status}" for name, status in statuses.items())
            + "\n"
            + (base / "summary.txt").read_text()
        )
        if view != Path():
            text += f"\nFiltered continuation analysis: at least {args.min_diversity_tokens} distinct {args.diversity_basis} clinical codes; raw results retained separately. See diversity_exclusions.csv for every denominator.\n"
        for cohort in ("random", "uncommon"):
            child_path = base / "cohorts" / cohort
            findings = pl.read_csv(
                child_path / "flagged_comparisons.csv",
                schema_overrides={
                    name: pl.Float64 for name in ("excess", "lower", "upper")
                },
            )
            text += f"\n{cohort}: {statuses[cohort]}; details: {child_path / 'summary.txt'}\n"
            for row in findings.iter_rows(named=True):
                text += f"  CHECK X=0 Y={row['Y']} {row['score']}: excess={100 * row['excess']:+.4f} pp [{100 * row['lower']:+.4f}, {100 * row['upper']:+.4f}]; n={row['patients_train']}/{row['patients_unseen']}.\n"
        (base / "summary.txt").write_text(text)
        (base / "report.txt").write_text(text)
        (base / "primary_summary.txt").write_text(text)
    write_audit(args.output_dir, args.output_dir / "audit.txt", args.min_unique_tokens)
    if args.plots:
        from bonsai.run.plot_memorization import plot_reports

        for directory in [
            args.output_dir / view / relative
            for view, _ in views
            for relative in (Path(), Path("cohorts/random"), Path("cohorts/uncommon"))
        ]:
            plot_reports(directory, directory / "plots")
    for directory in [
        args.output_dir / view / relative
        for view, _ in views
        for relative in (Path(), Path("cohorts/random"), Path("cohorts/uncommon"))
    ]:
        path = directory / "run_manifest.json"
        metadata = json.loads(path.read_text())
        metadata["completion_status"] = "complete"
        metadata["seconds"] = {"total": time.perf_counter() - started}
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    print((args.output_dir / "summary.txt").read_text().splitlines()[0], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--subject-id", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--evaluation-cache", type=Path)
    parser.add_argument("--reuse-generations", type=Path)
    parser.add_argument("--start-token", default="DOB")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-unique-tokens", type=int, default=3)
    parser.add_argument("--min-diversity-tokens", type=int)
    parser.add_argument(
        "--diversity-basis", choices=("reference", "predicted"), default="reference"
    )
    parser.add_argument("--unflagged-examples", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit_stored(args.audit_only, args.output_dir, args.subject_id)
        return
    if not args.report_dir or args.subject_id:
        parser.error("--report-dir is required; --subject-id belongs to --audit-only.")
    if (
        min(*args.horizons, args.batch_size, args.min_unique_tokens) < 1
        or args.unflagged_examples < 0
        or (args.max_patients is not None and args.max_patients < 2)
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
        or (args.min_diversity_tokens is not None and args.min_diversity_tokens < 1)
    ):
        parser.error("Positive grid/count/temperature/diversity settings required.")
    args.horizons = sorted(set(args.horizons))
    run(args)


if __name__ == "__main__":
    main()
