"""Read exported trajectory examples for a patient on the data server.

python -m bonsai.run.inspect_memorization --report-dir memorization --subject-id 123
Optional filters: --mode true_features --context 32 --horizon 32
For vim: omit --subject-id to include all exported examples and add
--output-file audit.txt --width 120, then open audit.txt in vim.
Output contains patient data. Only exported examples can be displayed: all
flagged trajectories and the saved sample of unflagged examples.
"""

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path


def examples(directory, subject_id, mode=None, context=None, horizon=None):
    diversity_filters = []
    analysis_cohort = None
    manifest_path = directory / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        diversity_filters = manifest.get("diversity_filters") or (
            [manifest["diversity_filter"]] if manifest.get("diversity_filter") else []
        )
        analysis_cohort = manifest.get("analysis_cohort")
        visited = {directory.resolve()}
        while source := manifest.get("reanalysis", {}).get("source_trajectory_exports"):
            directory = Path(source)
            if directory.resolve() in visited:
                raise ValueError(
                    "Cyclic trajectory-source provenance; cannot locate original exports."
                )
            visited.add(directory.resolve())
            manifest_path = directory / "run_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            )
    for name in ("flagged_trajectories.jsonl", "unflagged_examples.jsonl"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if analysis_cohort and row.get("sampling_cohort") != analysis_cohort:
                    continue
                excluded = False
                for diversity_filter in diversity_filters:
                    basis = diversity_filter["basis"]
                    count = row["scores"].get(f"unique_{basis}_tokens")
                    if count is None:
                        sequence = (
                            "true_continuation"
                            if basis == "reference"
                            else "predicted_continuation"
                        )
                        count = len(
                            {
                                token
                                for token in row[sequence]["tokens"]
                                if not token.startswith("[")
                            }
                        )
                    if count < diversity_filter["minimum_unique_tokens"]:
                        excluded = True
                        break
                if excluded:
                    continue
                if (
                    (subject_id is None or str(row["subject_id"]) == subject_id)
                    and (mode is None or row["mode"] == mode)
                    and (context is None or row["X"] == context)
                    and (horizon is None or row["Y"] == horizon)
                ):
                    yield row


def format_example(row, width=120, minimum_unique=3):
    """Plain text, bounded lines, no ANSI escapes or truncated token strings."""
    unique = row["scores"].get("unique_predicted_tokens")
    if unique is None:
        unique = len(
            {
                token
                for token in row["predicted_continuation"]["tokens"]
                if not token.startswith("[")
            }
        )
    lines = [
        "=" * width,
        f"Patient {row['subject_id']} | {row['split']} | {row['mode']} | cohort={row.get('sampling_cohort', 'all')} | X={row['X']} Y={row['Y']}",
        "Stored flags: " + (", ".join(row["flag_reasons"]) or "none at this cell"),
        f"Distinct predicted clinical codes: {unique} (overlap threshold: {minimum_unique})",
        f"Distinct reference clinical codes: {row['scores'].get('unique_reference_tokens', len({token for token in row['true_continuation']['tokens'] if not token.startswith('[')}))}",
    ]
    if seed := row.get("public_seed"):
        lines.extend(
            [
                f"Public model input: {seed['token']} (ID {seed['id']}); patient context=0; no patient features.",
                f"Stored assigned draw row: {row['generation_row']}; true continuation starts at original record offset {row['reference_offset']} after DOB.",
            ]
        )
    if selection := row.get("audit_selection"):
        lines.extend(
            [
                f"Selection: {selection['reason']} for group-level {selection['metric']} audit.",
                f"Original saved score={selection['saved_score']:.6f}; {'stored draw' if selection.get('stored_draw') else 'replay'} score={selection['replayed_score']:.6f}.",
                f"{'Stored draw' if selection.get('stored_draw') else 'Replay'} matches saved scores: {selection['replay_matches_saved_scores']}.",
            ]
        )
        if not selection["replay_matches_saved_scores"]:
            lines.append(
                "REPLAY DISAGREEMENT: this continuation is a new replay, not verified as the original prediction."
            )
    review_flags = []
    if row["Y"] >= 32:
        if row["scores"].get("exact_match") == 1:
            review_flags.append("exact_continuation")
        overlap = row["scores"].get("overlap")
        if overlap is not None and overlap >= 0.9 and unique >= minimum_unique:
            review_flags.append("overlap_ge_threshold")
    lines.append("Flags at selected threshold: " + (", ".join(review_flags) or "none"))
    if unique < minimum_unique:
        lines.append(
            "LOW DIVERSITY: repeated/simple continuation; not eligible for the current overlap trigger."
        )
        lines.append(
            "A perfect continuation can still trigger exact match. Original files remain unchanged."
        )
    for name in ("exact_match", "overlap", "rare_codes"):
        value = row["scores"].get(name)
        lines.append(
            f"  {name:14s}: {'not scored' if value is None else f'{100 * value:.2f}%'}"
        )
    if evidence := row.get("rare_code_evidence"):
        lines.extend(
            [
                "",
                "RARE-CODE MATCHES (unordered occurrence counts)",
                f"Matched {evidence['matched_rare_occurrences']}/{evidence['true_rare_occurrences']} true rare occurrences; generated rare occurrences={evidence['predicted_rare_occurrences']}.",
            ]
        )
        for code in evidence["codes"]:
            label = (
                "MATCHED"
                if code["matched_occurrences"]
                else "MISSED"
                if code["true_occurrences"]
                else "GENERATED ONLY"
            )
            lines.append(f"  {label}: {code['token']} (ID {code['code_id']})")
            lines.append(
                f"    true={code['true_occurrences']} predicted={code['predicted_occurrences']} matched={code['matched_occurrences']}"
            )
            if code["training_patient_count"] is not None:
                lines.append(
                    f"    Training holders: {code['training_patient_count']}/{code['training_patients']} ({100 * code['training_patient_fraction']:.6f}%)."
                )
            else:
                lines.append(
                    "    Training holders: unavailable; exact counts were not retained in this cache."
                )
    lines.extend(textwrap.wrap("History scope: " + row["scope"], width))

    def metadata(features, position):
        parts = []
        for field in ("age", "abspos", "segment", "numeric_value"):
            values = features.get(field)
            value = values[position] if values is not None else None
            if isinstance(value, float):
                value = f"{value:.6g}"
            parts.append(f"{field}={value if value is not None else 'missing'}")
        return "  ".join(parts)

    history = [
        "",
        "PUBLIC INPUT AND REFERENCE HISTORY {{{2"
        if row.get("public_seed")
        else "PROMPT HISTORY (true inputs) {{{2",
        "-" * width,
    ]
    history.extend(
        textwrap.wrap(
            "Decoded prompt: "
            + (
                row["public_seed"]["token"] + " (public token only)"
                if row.get("public_seed")
                else " | ".join(row["prompt"]["tokens"])
            ),
            width,
        )
    )
    for position, token in enumerate(row["prompt"]["tokens"]):
        history.extend(
            textwrap.wrap(f"{position:5d}  {token}", width, subsequent_indent="       ")
        )
        history.extend(
            textwrap.wrap(
                metadata(row["prompt_features"], position),
                width,
                initial_indent="       ",
                subsequent_indent="       ",
            )
        )
    if row.get("reference_history"):
        history.extend(
            [
                "",
                "REFERENCE HISTORY — NEVER MODEL INPUT",
                *textwrap.wrap(" | ".join(row["reference_history"]["tokens"]), width),
            ]
        )
    lines.extend(
        [
            "",
            "CONTINUATION (inspect this comparison first)",
            "-" * width,
            "= matching predicted code   ! mismatch   S supplied_SEP (not a prediction)",
            "Metadata below each row is the true reference, not generated metadata.",
        ]
    )
    column_width = (width - 12) // 2
    lines.append(f"{'offset':>6}  {'TRUE CODE':<{column_width}}  PREDICTED CODE")
    for position, (truth, prediction) in enumerate(
        zip(row["true_continuation"]["tokens"], row["predicted_continuation"]["tokens"])
    ):
        marker = (
            "S"
            if position in row["supplied_sep_positions"]
            else "="
            if truth == prediction
            else "!"
        )
        left = textwrap.wrap(truth, column_width) or [""]
        right = textwrap.wrap(prediction, column_width) or [""]
        for index in range(max(len(left), len(right))):
            prefix = f"{position:5d}{marker}" if index == 0 else "      "
            lines.append(
                f"{prefix}  {left[index] if index < len(left) else '':<{column_width}}  {right[index] if index < len(right) else ''}"
            )
        lines.extend(
            textwrap.wrap(
                metadata(row["true_continuation_features"], position),
                width,
                initial_indent="        ",
                subsequent_indent="        ",
            )
        )
    lines.extend(history)
    lines.append("}}}2")
    lines.extend(["", "LATER AVAILABLE REFERENCE (not generated) {{{2", "-" * width])
    for index, token in enumerate(
        row["later_available_reference"]["tokens"], row["X"] + row["Y"]
    ):
        lines.extend(
            textwrap.wrap(f"{index:5d}  {token}", width, subsequent_indent="       ")
        )
    if not row["later_available_reference"]["tokens"]:
        lines.append("None in this evaluated window.")
    lines.append("}}}2")
    return "\n".join(
        part
        for line in lines
        for part in (textwrap.wrap(line, width) if len(line) > width else [line])
    )


def write_audit(directory, output_file, minimum_unique=3, width=120):
    """Create an indexed, foldable audit; retain only compact index rows in RAM."""
    index, counts, patients = [], Counter(), set()
    for number, row in enumerate(examples(directory, None), 1):
        category = "FLAGGED" if row["flagged"] else "UNFLAGGED EXAMPLE"
        counts[category] += 1
        if row.get("audit_selection", {}).get("reason") == "rare_code_match_review":
            counts["RARE-CODE REVIEW"] += 1
            category = "RARE-CODE REVIEW"
        patients.add((row["split"], str(row["subject_id"])))
        unique = row["scores"].get("unique_predicted_tokens", "unknown")
        index.append(
            f"{number:06d}  Patient {row['subject_id']} | {row['split']} | {row['mode']} | X={row['X']} Y={row['Y']} | unique={unique} | {category}"
        )
    with output_file.open("x") as stream:
        header = [
            "BONSAI PATIENT AUDIT",
            "Contains patient data. Keep this file on the data server.",
            f"Exported cases: {len(index)}; distinct patients represented: {len(patients)}.",
            f"Individual reproduction flags: {counts['FLAGGED']}; no individual flag: {counts['UNFLAGGED EXAMPLE']}; rare-code review cases: {counts['RARE-CODE REVIEW']}.",
            f"Overlap diversity threshold: at least {minimum_unique} distinct predicted clinical codes.",
            "Cases are patient/context/horizon/mode combinations, not distinct patients.",
            "Flags suggest review, not proof of memorization. Low-diversity exact matches remain visible.",
            "Case groups reflect original export flags; each case also shows flags at the selected threshold.",
            "History is the evaluated window; it may be truncated. Unflagged examples are sampled.",
            "",
            "VIM NAVIGATION",
            "  :set foldmethod=marker foldlevel=1    Fold long history sections; keep comparisons visible.",
            "  zM / zR                             Close / open all folds.",
            "  za                                  Toggle the fold under the cursor.",
            "  /^CASE                              Find the next case; n continues.",
            "  /Patient 123                        Find a patient in the index or cases.",
            "  /^CASE 000123                        Jump directly to indexed case 000123.",
            "  /LOW DIVERSITY                      Find simple/repetitive continuations.",
            "  /REVIEW NOTE                        Find the blank note area in each case.",
            "",
            "CASE INDEX (individual flags first, then other exported review examples)",
        ]
        manifest_path = directory / "run_manifest.json"
        metadata = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        if cell := metadata.get("cell_audit"):
            header[1:1] = [
                f"GROUP-LEVEL CELL AUDIT: {cell['mode']} X={cell['X']} Y={cell['Y']} metric={cell['metric']} cohort={cell['cohort']}",
                "Selection: "
                + cell["selection"]
                + ". These examples are not population estimates.",
                "Trajectories are regenerated; saved-score agreement checks consistency, not token-identical original predictions.",
                f"Replay mismatches: {cell['replay_mismatches']}; check original/replayed score labels in each case.",
                "Read summary.txt for the distribution of ALL contributing patients and the saved-score excess.",
            ]
        for line in header + index:
            stream.write("\n".join(textwrap.wrap(line, width) or [""]) + "\n")
        for number, row in enumerate(examples(directory, None), 1):
            category = "FLAGGED AT EXPORT" if row["flagged"] else "UNFLAGGED EXAMPLE"
            reason = row.get("audit_selection", {}).get("reason")
            if reason == "rare_code_match_review":
                category = "RARE-CODE REVIEW"
            elif reason == "zero_score_control":
                category = "ZERO-SCORE CONTROL"
            stream.write(
                f"\nCASE {number:06d} | Patient {row['subject_id']} | {category} "
                + "{{{1\n"
            )
            stream.write(format_example(row, width, minimum_unique) + "\n")
            stream.write(
                "\nREVIEW NOTE (manual; does not change computed report status):\n  Decision:\n  Reason:\n  Reviewer/date:\n}}}1\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--subject-id",
        help="Omit to format all exported flagged and unflagged examples",
    )
    parser.add_argument(
        "--output-file", type=Path, help="Write a new plain-text audit file for vim"
    )
    parser.add_argument(
        "--min-unique-overlap-tokens",
        type=int,
        help="Override diversity threshold; otherwise use the report's setting",
    )
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--mode")
    parser.add_argument("--context", type=int)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    if args.min_unique_overlap_tokens is None:
        manifest_path = args.report_dir / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        args.min_unique_overlap_tokens = manifest.get("inspection_triggers", {}).get(
            "minimum_unique_predicted_clinical_tokens_for_overlap", 3
        )
    if args.min_unique_overlap_tokens < 1:
        parser.error("--min-unique-overlap-tokens must be positive.")
    if args.width < 80:
        parser.error("--width must be at least 80.")
    if args.output_file and not any(
        (args.subject_id, args.mode, args.context, args.horizon)
    ):
        write_audit(
            args.report_dir,
            args.output_file,
            args.min_unique_overlap_tokens,
            args.width,
        )
        return
    found = False
    stream = args.output_file.open("x") if args.output_file else None
    for row in examples(
        args.report_dir, args.subject_id, args.mode, args.context, args.horizon
    ):
        print(
            format_example(row, args.width, args.min_unique_overlap_tokens) + "\n",
            file=stream,
        )
        found = True
    if stream is not None:
        stream.close()
    if not found:
        parser.exit(
            1,
            "No saved example matches. Inspect patient_scores.csv for IDs; unflagged trajectories are sampled, and older runs need regeneration with --export-patient-inspection.\n",
        )


if __name__ == "__main__":
    main()
