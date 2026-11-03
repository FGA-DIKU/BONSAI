"""Read exported trajectory examples for a patient on the data server.

python -m bonsai.run.inspect_memorization --report-dir memorization --subject-id 123
Optional filters: --mode true_features --context 32 --horizon 32
Output contains patient data. Only exported examples can be displayed: all
flagged trajectories and the saved sample of unflagged examples.
"""

import argparse
import json
from pathlib import Path


def examples(directory, subject_id, mode=None, context=None, horizon=None):
    for name in ("flagged_trajectories.jsonl", "unflagged_examples.jsonl"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    str(row["subject_id"]) == subject_id
                    and (mode is None or row["mode"] == mode)
                    and (context is None or row["X"] == context)
                    and (horizon is None or row["Y"] == horizon)
                ):
                    yield row


def format_example(row):
    lines = [
        f"Patient {row['subject_id']} | {row['split']} | {row['mode']} | X={row['X']} Y={row['Y']}",
        f"Flags: {', '.join(row['flag_reasons']) or 'none at this cell'}",
        f"Scores: {row['scores']}",
        f"History scope: {row['scope']}",
        "Prompt history (true inputs):",
        "position\tcode\tage\tabspos\tsegment\tnumeric_value",
    ]
    for position, token in enumerate(row["prompt"]["tokens"]):
        values = [
            row["prompt_features"].get(field, [None] * row["X"])[position]
            for field in ("age", "abspos", "segment", "numeric_value")
        ]
        lines.append("\t".join(map(str, [position, token, *values])))
    lines.extend(
        [
            "Continuation (metadata columns are reference values):",
            "offset\ttrue_code\tpredicted_code\tsupplied_SEP\tage\tabspos\tsegment\tnumeric_value",
        ]
    )
    for position, (truth, prediction) in enumerate(
        zip(row["true_continuation"]["tokens"], row["predicted_continuation"]["tokens"])
    ):
        values = [
            row["true_continuation_features"].get(field, [None] * row["Y"])[position]
            for field in ("age", "abspos", "segment", "numeric_value")
        ]
        lines.append(
            "\t".join(
                map(
                    str,
                    [
                        position,
                        truth,
                        prediction,
                        position in row["supplied_sep_positions"],
                        *values,
                    ],
                )
            )
        )
    lines.append(
        "Later available reference: "
        + " | ".join(row["later_available_reference"]["tokens"])
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--mode")
    parser.add_argument("--context", type=int)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    found = False
    for row in examples(
        args.report_dir, args.subject_id, args.mode, args.context, args.horizon
    ):
        print(format_example(row) + "\n")
        found = True
    if not found:
        parser.exit(
            1,
            "No saved example matches. Inspect patient_scores.csv for IDs; unflagged trajectories are sampled, and older runs need regeneration with --export-patient-inspection.\n",
        )


if __name__ == "__main__":
    main()
