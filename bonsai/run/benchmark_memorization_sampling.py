"""Synthetic scale benchmark of rarity selection; NOT a real-population benchmark.

python -m bonsai.run.benchmark_memorization_sampling --output /tmp/sampling.json
To measure real source loading/selection, run memorization_check with
--prepare-sampling-only --device cpu and the real checkpoint/data/config paths.
"""

import argparse
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from bonsai.functional.memorization_sampling import SamplingSettings, select_cohorts


def benchmark(settings, vocab_size=100000, distribution="zipf"):
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rng = np.random.default_rng(42)
    sizes = {
        "train": settings.reference_patients
        + settings.calibration_patients
        + settings.candidate_limit
        + settings.patients_per_split,
        "tuning": settings.candidate_limit + settings.patients_per_split,
    }
    lengths = {
        split: rng.choice([96, 320, 1200, 8193], count)
        for split, count in sizes.items()
    }

    def read_codes(split, index, stop):
        local = np.random.default_rng(
            np.random.SeedSequence([42, 0 if split == "train" else 1, index])
        )
        codes = (
            np.minimum(local.zipf(1.5, 1100), vocab_size - 1)
            if distribution == "zipf"
            else local.integers(1, vocab_size, 1100)
        )
        rare = local.random(1100) < 0.05
        codes[rare] = local.integers(1, vocab_size, rare.sum())
        return np.resize(codes, min(stop, int(lengths[split][index])))

    started = time.perf_counter()
    selected, audit = select_cohorts(
        lengths, read_codes, set(range(1, vocab_size)), [32, 256, 1024], 33, settings
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024**2 if sys.platform == "darwin" else 1024
    audit.pop("reference_indices")
    audit.pop("calibration_indices")
    return {
        "scope": "SYNTHETIC; excludes torch.load, metadata deserialization, validation, global rare-code counts, cache serialization and model inference",
        "vocab_size": vocab_size,
        "synthetic_distribution": distribution,
        "population_sizes": sizes,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": peak / scale,
        "peak_rss_increase_mib": (peak - baseline) / scale,
        "selected": {split: len(rows) for split, rows in selected.items()},
        "sampling": audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-patients", type=int, default=20000)
    parser.add_argument("--calibration-patients", type=int, default=5000)
    parser.add_argument("--candidate-limit", type=int, default=20000)
    parser.add_argument("--vocab-size", type=int, default=100000)
    parser.add_argument("--distribution", choices=("zipf", "uniform"), default="zipf")
    args = parser.parse_args()
    settings = replace(
        SamplingSettings(),
        reference_patients=args.reference_patients,
        calibration_patients=args.calibration_patients,
        candidate_limit=args.candidate_limit,
    )
    result = benchmark(settings, args.vocab_size, args.distribution)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sampling"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
