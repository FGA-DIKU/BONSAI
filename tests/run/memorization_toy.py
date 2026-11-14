"""End-to-end synthetic check: train a tiny BONSAI, save checkpoints, run the real CLI.

All records contain synthetic toy tokens, never real patients. The learned
model fits eight distinct repeating code patterns. A zero-weight model is the control.
The same learned sequences in both groups test the 'common/predictable' caveat.

    python -m tests.run.memorization_toy --output-dir /tmp/bonsai_memorization_toy
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bonsai.modules.datasets.PretrainDataset import ARPretrainDataset
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain, BonsaiValuePretrain


def records(rng, offset, patterns=False):
    """Distinct synthetic patients, each with 64 non-padding code tokens."""
    return [
        {
            "subject_id": offset + i,
            "code": torch.tensor(
                np.resize(np.arange(1 + 3 * i, 4 + 3 * i), 64)
                if patterns
                else rng.integers(1, 25, size=64)
            ),
            "age": torch.zeros(64),
            "abspos": torch.arange(64).float(),
            "segment": torch.ones(64, dtype=torch.long),
            "numeric_value": torch.full((64,), 0.5),
        }
        for i in range(8)
    ]


def save_checkpoint(network, path):
    torch.save(
        {
            "hyper_parameters": network.hparams,
            "state_dict": {
                f"model.{key}": value for key, value in network.state_dict().items()
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    train, tuning = records(rng, 0, patterns=True), records(rng, 100)
    network = BonsaiPretrain(
        vocab_size=25,
        max_seqlen=128,
        hidden_size=64,
        num_layers=1,
        bias=False,
        dropout=0.0,
        attention_dropout=0.0,
        causal=True,
        attn_type="sdpa",
    )
    dataset = ARPretrainDataset(train, max_len=128, background_length=64)
    examples = [dataset[i] for i in range(len(dataset))]
    batch = {
        key: torch.stack([row[key] for row in examples])
        for key in ("code", "age", "abspos", "segment", "attention_mask", "target")
    }
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.005)
    network.train()
    for step in range(1000):
        optimizer.zero_grad()
        logits, labels = network(batch)
        token_loss = F.cross_entropy(logits, labels, reduction="none")
        loss = token_loss.mean()
        loss.backward()
        optimizer.step()
        if token_loss.max().item() < 0.1:
            break
    else:
        raise AssertionError(
            "Toy did not learn all next-code transitions within the training budget"
        )
    print(
        f"Toy fitted in {step + 1} steps; max teacher-forced token CE={token_loss.max().item():.4f}",
        flush=True,
    )
    network.eval()
    trained = root / "memorizing.ckpt"
    save_checkpoint(network, trained)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    control = root / "uninformative.ckpt"
    save_checkpoint(network, control)

    value_network = BonsaiValuePretrain(
        **{**network.hparams, "attention_dropout": 0.0, "value_embedding_mode": "film"}
    )
    with torch.no_grad():
        for parameter in value_network.parameters():
            parameter.zero_()
    value_control = root / "value_control.ckpt"
    save_checkpoint(value_network, value_control)

    outcomes = {}
    for name, checkpoint, comparison, minimum, expected in (
        ("memorizing", trained, tuning, 4, "CHECK:"),
        ("uninformative", control, tuning, 4, "INCONCLUSIVE:"),
        ("value_model_control", value_control, tuning, 4, "INCONCLUSIVE:"),
        ("missing_run_settings", control, tuning, 4, "INCONCLUSIVE:"),
        ("unsupported_grid", control, tuning, 4, "INCONCLUSIVE:"),
        (
            "common_sequences",
            trained,
            [{**row, "subject_id": row["subject_id"] + 100} for row in train],
            4,
            "INCONCLUSIVE:",
        ),
        ("insufficient_sample", control, tuning, 50, "INCONCLUSIVE:"),
    ):
        data = root / name
        data.mkdir()
        torch.save(train, data / "subject_data_train.pt")
        torch.save(comparison, data / "subject_data_tuning.pt")
        torch.save(
            {"[PAD]": 0, **{f"TOY_{i}": i for i in range(1, 25)}},
            data / "vocabulary.pt",
        )
        population = data / "population.csv"
        population.write_text(
            "subject_id\n"
            + "\n".join(str(row["subject_id"]) for row in train + comparison)
            + "\n"
        )
        config = data / "config.yaml"
        config.write_text(
            f"paths:\n  population: {population}\ntraining:\n  max_len: 128\n  cutoff_date: null\n"
        )
        output = data / "report"
        command = [
            sys.executable,
            "-m",
            "bonsai.run.memorization_check",
            "--sampling-design",
            "legacy",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data),
            "--run-config",
            str(config),
            "--output-dir",
            str(output),
            "--prompt-events",
            "16",
            "32",
            "--continuation-events",
            "8",
            "16",
            "32",
            "--max-patients",
            "8",
            "--min-patients",
            str(minimum),
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--data-version",
            "synthetic-seed-7",
        ]
        if name == "missing_run_settings":
            index = command.index("--run-config")
            del command[index : index + 2]
        if name == "unsupported_grid":
            command.extend(["--prompt-events", "16", "32", "256"])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (data / "stdout.txt").write_text(result.stdout)
        summary = (output / "summary.txt").read_text()
        assert summary.startswith(expected), summary
        manifest = json.loads((output / "run_manifest.json").read_text())
        assert manifest["status"] == "EXPLORATORY_ONLY"
        assert set(manifest["generation_modes"]) == {
            "code_only",
            "true_features",
            "forward_fill",
        }
        assert manifest["evidence_summary"] == summary.splitlines()[0]
        outcomes[name] = summary.splitlines()[0]
        print(f"{name}: {outcomes[name]}", flush=True)
    (root / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    print(f"Actual CLI reports saved to {root}")


if __name__ == "__main__":
    main()
