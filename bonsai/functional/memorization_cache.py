"""Local patient-window cache for repeated memorization diagnostics.

Contains patient data: keep it on the data server, separate from model bundles.
Only sampled, bounded windows are stored. Full-training rarity groups are retained
separately so their definition does not change when reusing the sample.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch


class CachedWindows:
    """Deduplicated windows with preselected generation and loss pools."""

    def __init__(self, payload):
        self.rows = payload["rows"]
        self.pools = payload["pools"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def sample(self, min_events, maximum, require_targets):
        key = "loss" if require_targets else min_events
        indices = self.pools[key]
        # A fixed random ordering gives nested, uniform subsets for preflight/full.
        order = np.random.default_rng(0).permutation(len(indices))[:maximum]
        return [self.rows[indices[int(index)]] for index in sorted(order)]


class FixedCohortWindows(CachedWindows):
    """One fixed ID set across prompts and loss; filter eligibility after selection."""

    def __init__(self, payload):
        self.rows = payload["rows"]
        self.members = payload["members"]
        self.random_fraction = payload["random_fraction"]
        self.excluded_target = payload["excluded_target"]

    def sample(self, min_events, maximum, require_targets):
        chosen = []
        for cohort in ("random", "uncommon"):
            indices = [
                i for i, row in enumerate(self.rows) if row["sampling_cohort"] == cohort
            ]
            target = (
                round(maximum * self.random_fraction)
                if cohort == "random"
                else maximum - round(maximum * self.random_fraction)
            )
            order = np.random.default_rng(0).permutation(len(indices))[:target]
            chosen.extend(indices[int(i)] for i in order)
        result = []
        for index in sorted(chosen):
            row = self.rows[index]
            if len(row["code"]) < min_events:
                continue
            if require_targets:
                valid = row["code"][1:] != 0
                if self.excluded_target is not None:
                    valid &= row["code"][1:] != self.excluded_target
                if not bool(valid.any()):
                    continue
            result.append(row)
        return result


def source_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_pools(
    windows, minimum_lengths, capacity, sampler, excluded_target, generation_limit=None
):
    """Use the original full-population sampler, deduplicating bounded windows."""
    result = {}
    for split, records in windows.items():
        rows, by_id, pools = [], {}, {}
        requests = [(length, False) for length in minimum_lengths] + [(2, True)]
        for length, loss in requests:
            selected = sampler(
                records,
                length,
                capacity,
                require_targets=loss,
                exclude_target=excluded_target if loss else None,
            )
            indices = []
            for row in selected:
                identifier = row["subject_id"]
                if not loss and generation_limit is not None:
                    # Clone slices so torch.save does not retain full backing storage.
                    row = {
                        key: value[:generation_limit].clone()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in row.items()
                    }
                if identifier not in by_id:
                    by_id[identifier] = len(rows)
                    rows.append(row)
                elif len(row["code"]) > len(rows[by_id[identifier]]["code"]):
                    rows[by_id[identifier]] = row
                indices.append(by_id[identifier])
            pools["loss" if loss else length] = indices
        result[split] = {"rows": rows, "pools": pools}
    return result


def save_cache(path, signature, capacity, pools, countable, rare, sampling=None):
    """Atomic publication; interrupted preparation leaves no apparently valid cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(
            {
                "version": 2 if sampling is not None else 1,
                "signature": signature,
                "capacity": capacity,
                "pools": pools,
                "countable": countable,
                "rare": rare,
                "sampling": sampling,
            },
            temporary,
        )
        os.link(temporary, path)  # Publish without overwriting a concurrent writer.
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_cache(path, signature, requested):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("version") not in (1, 2) or payload.get("signature") != signature:
        raise ValueError(
            "Evaluation cache settings/source files differ. Use a new cache path."
        )
    if payload["capacity"] < requested:
        raise ValueError(
            "Evaluation cache capacity is too small. Use a new cache path with a larger --cache-patients."
        )
    windows = {
        split: (FixedCohortWindows(pool) if "members" in pool else CachedWindows(pool))
        for split, pool in payload["pools"].items()
    }
    for dataset in windows.values():
        dataset.sampling_audit = payload.get("sampling")
    return (
        windows,
        payload["countable"],
        payload["rare"],
        payload["capacity"],
    )
