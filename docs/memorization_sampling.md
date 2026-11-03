The default `fixed` design evaluates 10,000 distinct patients: 4,000 random and
1,000 additional uncommon patients from each of train and tuning. IDs stay fixed
across all contexts, horizons, generation modes and the loss comparison. Short
histories contribute only to eligible cells; they are not replaced.

The rarity scorer never reads BONSAI predictions or weights. A patient-weighted
code-pair frequency reference is fitted on 20,000 uniformly reserved training
patients. Repeated pairs count once per reference patient. A separate 5,000
reserved training patients calibrate the maximum average surprisal over eligible
32-position spans after prompt boundaries 32, 256 and 1,024. Special-token targets
are excluded; each span needs eight clinical positions. Smoothing backs unseen
pairs off toward reference unigram frequencies.

Both reserved groups are disjoint from evaluated training patients. Cutoffs are
calibrated separately by which boundaries were usable, accounting for the maximum
over multiple boundaries. The uncommon sample comes from the upper 10% reference
tail, with shared train/tuning quotas by boundary availability and rarity bands
(90–95%, 95–99%, and the uppermost band where distinguishable). These are
calibration bands, not BONSAI performance percentiles. Score ties are explicit in
the audit; completely tied calibration scores do not define an uncommon tail.

Candidates are visited in seeded random order, stopping at the fixed allocations
or 100,000 candidates per split. Stopping never uses reproduction outcomes.
Shortages or unsupported calibration strata prevent an overall pass. This is a
transparent short-transition rarity heuristic; it does not establish clinical
unusualness or identify every rare long-range sequence.

The random cohort provides the population-facing comparison. Uncommon and
combined cohorts are deliberately enriched stress samples. Coarse matching does
not remove every demographic or history-composition difference; inspect the
balance output. Intervals are pointwise screening intervals, not simultaneous
equivalence bounds. Individual exact/overlap triggers remain audit evidence.

## What reads the full population?

- The current monolithic `.pt` files require deserializing all Python metadata.
- Cutoff/window lengths and duplicate/split-overlap checks cover the population.
- The existing rare-code reproduction metric retains its definition from all
  training windows. One code-only frequency pass is still required initially.
- New sequence-rarity fitting, calibration and candidate scoring use only bounded
  prefixes of the reserved/sampled candidates, not a population-wide BONSAI pass.
- Full input-feature validation and inference use selected evaluated windows.
- The fixed cache stores the selected windows, selection provenance and full-
  population rare-code groups. Later runs skip population loading and counting.

Changing sampling settings requires a new cache; legacy per-prompt caches cannot
provide the independent reference or additional uncommon patients. Use
`--sampling-design legacy` only to reproduce the earlier design.

## Measured local benchmark

No actual server population or processed patient files were available locally.
The following are **synthetic CPU benchmarks**, with 100,000 vocabulary entries,
50,000/25,000 candidate-population sizes, 20,000 fitting patients, 5,000 calibration
patients and 5,000 evaluated patients per split. Both filled all matched quotas.

| Synthetic code distribution | Selection wall time | Peak RSS growth | Distinct reference pairs |
|---|---:|---:|---:|
| Zipf with rare-code noise | 12.7 s | 55 MiB | 214,212 |
| Uniform, more varied transitions | 8.4 s | 225 MiB | 1,440,914 |

These measurements include synthetic prefix generation, fitting, calibration and
candidate selection. They exclude PyTorch/model memory, source deserialization,
full-training metric-frequency counting, feature validation, cache writing and
model inference. They are not estimates of full server startup or run time.

Reproduce the synthetic stress benchmark:

```bash
python -m bonsai.run.benchmark_memorization_sampling \
  --distribution uniform --output /tmp/sampling_benchmark.json
```

Measure actual population preparation without inference:

```bash
python -u -m bonsai.run.memorization_check \
  --checkpoint /path/to/best.ckpt \
  --run-config /path/to/run/hydra/config.yaml \
  --data-dir /path/to/processed/data \
  --evaluation-cache /local/path/memorization_fixed_v1.pt \
  --output-dir /path/to/new/sampling_preparation \
  --sampling-design fixed --prepare-sampling-only \
  --sep-excluded-from-training-loss --device cpu
```

Preserve the training population/cutoff/context/data-version overrides. For the
full run use the same cache and settings, a fresh output directory, omit
`--prepare-sampling-only`, select CUDA/batch size, and add
`--export-patient-inspection`. Preparation and full inference can also happen in
one run by omitting the preparation flag from the start.

Preparation writes `sampling_report.json` (including separate loading,
reference/calibration/selection and full-frequency timings),
`sampling_patients.csv` (local patient IDs and selection scores) and
`sampling_balance.csv`. These local artifacts contain patient information.
Reports in the root directory describe the combined sample; `cohorts/random/`
and `cohorts/uncommon/` report the same inference separately. The root verdict
combines all three statuses. Missing rare-code score denominators are reported
as inconclusive, rather than treated as passing.
