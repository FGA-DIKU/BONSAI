"""Fixed random/uncommon cohorts selected without BONSAI predictions.

Fit and calibration patients are independent random training subsets, excluded
from evaluated training IDs. A smoothed, patient-weighted code-pair reference
scores short spans at fixed prompt boundaries. Repeated transitions count once
per reference patient. This is a sampling heuristic, not a privacy guarantee.
"""

import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingSettings:
    patients_per_split: int = 100000
    random_fraction: float = 0.8
    reference_patients: int = 100000
    calibration_patients: int = 20000
    span: int = 32
    minimum_clinical_tokens: int = 8
    quantile: float = 0.9
    smoothing: float = 10.0
    minimum_calibration: int = 50
    candidate_limit: int = 1000000
    seed: int = 0

    def validate(self):
        if self.patients_per_split < 2 or not 0 < self.random_fraction < 1:
            raise ValueError(
                "Fixed cohorts need >=2 patients per split and random fraction between 0 and 1."
            )
        if (
            min(
                self.reference_patients,
                self.calibration_patients,
                self.span,
                self.minimum_clinical_tokens,
                self.minimum_calibration,
                self.candidate_limit,
            )
            < 1
        ):
            raise ValueError(
                "Rarity reference/calibration/span/candidate settings must be positive."
            )
        if (
            self.minimum_clinical_tokens > self.span
            or not 0 < self.quantile < 1
            or self.smoothing <= 0
        ):
            raise ValueError("Invalid rarity token count, quantile or smoothing.")
        random_count = round(self.patients_per_split * self.random_fraction)
        if min(random_count, self.patients_per_split - random_count) < 2:
            raise ValueError(
                "Both random and uncommon allocations need at least two patients."
            )


class SequenceReference:
    def __init__(self, clinical_codes, prompts, settings):
        self.clinical = set(clinical_codes)
        self.prompts = tuple(prompts)
        self.settings = settings
        self.radix = max(self.clinical, default=0) + 1
        self.pairs, self.prefixes, self.tokens = Counter(), Counter(), Counter()
        self.observations = 0

    def blocks(self, codes):
        for bit, prompt in enumerate(self.prompts):
            if len(codes) < prompt + self.settings.span:
                continue
            pairs = [
                (int(codes[i - 1]), int(codes[i]))
                for i in range(prompt, prompt + self.settings.span)
                if int(codes[i]) in self.clinical
            ]
            if len(pairs) >= self.settings.minimum_clinical_tokens:
                yield 1 << bit, pairs

    def fit(self, histories):
        for codes in histories:
            pairs = {pair for _, block in self.blocks(codes) for pair in block}
            self.observations += len(pairs)
            self.pairs.update(a * self.radix + b for a, b in pairs)
            self.prefixes.update(a for a, _ in pairs)
            self.tokens.update({b for _, b in pairs})
        self.total_tokens = sum(self.tokens.values())
        if not self.pairs:
            raise ValueError(
                "Reference has no usable clinical transitions. Adjust prompts/span or reference size."
            )

    def score(self, codes):
        mask, scores = 0, []
        denominator = self.total_tokens + len(self.clinical)
        alpha = self.settings.smoothing
        for bit, pairs in self.blocks(codes):
            # Unique transitions prevent a repeated medication/SEP pair dominating.
            surprises = []
            for a, b in set(pairs):
                unigram = (self.tokens[b] + 1) / denominator
                probability = (self.pairs[a * self.radix + b] + alpha * unigram) / (
                    self.prefixes[a] + alpha
                )
                surprises.append(-math.log(probability))
            mask |= bit
            scores.append(float(np.mean(surprises)))
        # Calibration is stratified by this mask, accounting for how many and
        # which boundaries had usable spans; longer histories get no free looks.
        return mask, max(scores) if scores else math.nan


def _allocate(total, sizes):
    keys = sorted(sizes)
    weights = np.array([sizes[key] for key in keys], dtype=float)
    expected = total * weights / weights.sum()
    allocation = np.floor(expected).astype(int)
    for index in np.argsort(-(expected - allocation), kind="stable")[
        : total - allocation.sum()
    ]:
        allocation[index] += 1
    return dict(zip(keys, map(int, allocation)))


def select_cohorts(
    lengths_by_split,
    read_codes,
    clinical_codes,
    prompts,
    minimum_length,
    settings,
    excluded_target=None,
):
    """Return fixed source indices plus diagnostics; never replace short grid cases.

    read_codes(split, index, stop) must return the SAME bounded/censored training
    window prefix used for inference. Only reference/calibration/candidate code
    prefixes are read; other input features are not materialized here.
    """
    settings.validate()
    started = time.perf_counter()
    splits = tuple(lengths_by_split)
    train, _tuning = splits
    rng = np.random.default_rng(settings.seed)
    eligible = {
        split: rng.permutation(np.flatnonzero(np.asarray(lengths) >= minimum_length))
        for split, lengths in lengths_by_split.items()
    }
    reserved = settings.reference_patients + settings.calibration_patients
    if len(eligible[train]) < reserved + settings.patients_per_split:
        raise ValueError(
            "Not enough eligible training patients for independent reference/calibration and evaluation. Reduce the explicit sampling sizes."
        )
    fit_ids = eligible[train][: settings.reference_patients]
    calibration_ids = eligible[train][settings.reference_patients : reserved]
    eligible[train] = eligible[train][reserved:]
    stop = max(prompts) + settings.span
    reference = SequenceReference(clinical_codes, prompts, settings)
    print(
        f"Fitting rarity reference: {len(fit_ids):,} training patients (code prefixes only)",
        flush=True,
    )
    reference.fit(read_codes(train, int(i), stop) for i in fit_ids)
    fitted = time.perf_counter()
    calibration = defaultdict(list)
    for i in calibration_ids:
        mask, score = reference.score(read_codes(train, int(i), stop))
        if mask:
            calibration[mask].append(score)
    usable = {
        mask: values
        for mask, values in calibration.items()
        if len(values) >= settings.minimum_calibration
    }
    if not usable:
        raise ValueError(
            "Too few independent calibration patients per boundary stratum. Increase calibration size or simplify the grid."
        )
    cutoffs = {
        mask: float(np.quantile(values, settings.quantile))
        for mask, values in usable.items()
    }
    inclusive = {
        mask: cutoffs[mask] > float(np.median(values)) + 1e-12
        for mask, values in usable.items()
    }
    random_count = round(settings.patients_per_split * settings.random_fraction)
    uncommon_count = settings.patients_per_split - random_count
    band_edges = {
        mask: [
            float(
                np.quantile(
                    values, settings.quantile + fraction * (1 - settings.quantile)
                )
            )
            for fraction in (0.5, 0.9)
        ]
        for mask, values in usable.items()
    }

    def stratum(mask, score):
        return (
            mask * 10 + int(np.searchsorted(band_edges[mask], score, side="right"))
            if mask in band_edges
            else 0
        )

    def qualifies(mask, score):
        return mask in cutoffs and (
            score >= cutoffs[mask] if inclusive[mask] else score > cutoffs[mask]
        )

    tail_sizes = Counter(
        stratum(mask, score)
        for mask, values in usable.items()
        for score in values
        if qualifies(mask, score)
    )
    if not tail_sizes:
        raise ValueError(
            "Calibration has no distinct upper rarity tail; refusing to invent an uncommon cohort."
        )
    quotas = _allocate(uncommon_count, tail_sizes)
    calibrated = time.perf_counter()
    selected, diagnostics = {}, {}
    for split in splits:
        random_ids = []
        # Random assignment uses only length/target eligibility, never surprise.
        for position, index in enumerate(eligible[split]):
            index = int(index)
            if len(random_ids) < random_count:
                codes = np.asarray(
                    read_codes(split, index, int(lengths_by_split[split][index]))
                )
                valid = codes[1:] != 0
                if excluded_target is not None:
                    valid &= codes[1:] != excluded_target
                if valid.any():
                    random_ids.append(index)
                    if len(random_ids) == random_count:
                        break
        if len(random_ids) != random_count:
            raise ValueError(f"Too few target-eligible random patients in {split}.")
        remaining = eligible[split][position + 1 :]
        members = []
        for index in random_ids:
            mask, score = reference.score(read_codes(split, index, stop))
            members.append(
                {
                    "index": index,
                    "cohort": "random",
                    "rarity_mask": mask,
                    "rarity_score": score,
                    "rarity_stratum": stratum(mask, score),
                    "rarity_cutoff": cutoffs.get(mask),
                }
            )
        found, inspected = Counter(), 0
        for index in remaining:
            index = int(index)
            if (
                all(found[mask] >= quota for mask, quota in quotas.items())
                or inspected >= settings.candidate_limit
            ):
                break
            inspected += 1
            mask, score = reference.score(read_codes(split, index, stop))
            cell = stratum(mask, score)
            if cell in quotas and found[cell] < quotas[cell] and qualifies(mask, score):
                members.append(
                    {
                        "index": index,
                        "cohort": "uncommon",
                        "rarity_mask": mask,
                        "rarity_score": score,
                        "rarity_stratum": cell,
                        "rarity_cutoff": cutoffs[mask],
                    }
                )
                found[cell] += 1
        selected[split] = members
        diagnostics[split] = {
            "random": len(random_ids),
            "uncommon": sum(found.values()),
            "uncommon_requested": uncommon_count,
            "candidates_scored": inspected,
            "uncommon_by_stratum": dict(found),
            "quota_by_stratum": quotas,
            "complete": all(found[mask] == quota for mask, quota in quotas.items()),
        }
        print(
            f"Selected {split}: random={len(random_ids):,}, uncommon={sum(found.values()):,}/{uncommon_count:,}; candidate prefixes scored={inspected:,}",
            flush=True,
        )
    finished = time.perf_counter()
    audit = {
        "settings": asdict(settings),
        "prompts": list(prompts),
        "reference_indices": list(map(int, fit_ids)),
        "calibration_indices": list(map(int, calibration_ids)),
        "pair_entries": len(reference.pairs),
        "reference_transition_observations": reference.observations,
        "calibration_counts": {
            mask: len(values) for mask, values in calibration.items()
        },
        "cutoffs": cutoffs,
        "rarity_band_edges": band_edges,
        "calibration_tail_counts_by_stratum": dict(tail_sizes),
        "splits": diagnostics,
        "cutoff_inclusive": inclusive,
        "calibration_tail_fractions": {
            mask: float(
                np.mean(
                    np.array(values) >= cutoffs[mask]
                    if inclusive[mask]
                    else np.array(values) > cutoffs[mask]
                )
            )
            for mask, values in usable.items()
        },
        "timing_seconds": {
            "reference_fit": fitted - started,
            "calibration": calibrated - fitted,
            "selection": finished - calibrated,
            "total": finished - started,
        },
        "definition": "max smoothed patient-weighted transition surprisal across eligible boundaries; common cutoffs and matched quotas by boundary availability and rarity band; no BONSAI predictions",
        "calibration_reliable": all(
            len(v) >= settings.minimum_calibration for v in calibration.values()
        ),
    }
    return selected, audit
