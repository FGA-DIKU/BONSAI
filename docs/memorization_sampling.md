## Supplemental prompted analyses and readable reports

The existing context-conditioned, greedy analyses are supplemental to the
[context-zero per-patient comparison](memorization_zero_context.md).
Their results and metrics are retained.
New inference and reanalysis runs automatically write `report.txt`,
`flagged_comparisons.csv` and `flagged_loss_checks.csv`. The text distinguishes
supplemental statuses from the separate primary analysis. Add `--primary-report`
to include its completed results before the supplemental results.

Generate the report and figures from **existing** results, without patient/model
loading or inference:

```bash
pip install '.[plots]'
python -m bonsai.run.report_memorization --report-dir /path/to/results
vim /path/to/results/report.txt
```

Use `--no-plots` for text/CSV only, or `--output-dir /path/to/new/report` to keep
all derived artifacts outside the source results. Known random/uncommon and
diversity-filtered folders are included automatically. Figures show train/tuning
mean scores and excess intervals over horizons, exact/overlap patient rates,
loss AUC, and cohort/mode comparison grids. Circles on the grids mark assessable
pointwise intervals excluding zero in either direction. Missing/underpowered
cells are grey; plot colors express percentage points, not significance.

Additional diagnostic figures show actual per-metric denominators (particularly
the smaller rare-code denominators), coverage gaps, median/p95/max patient scores,
exact-match counts split by diversity, overall low-diversity rates, and retention
under the diversity filter. Exact-match panels share a vertical scale within each
figure and annotate exact/evaluated and low-diversity exact/exact counts. These
descriptive plots do not add statistical tests or change the saved verdicts.

Optionally add `--include-patient-distributions` to plot histograms and empirical
quantile curves of saved teacher-forced losses. This reads only split/loss columns
from `patient_losses.csv`, without IDs or trajectories, and requires no inference.
Absent optional inputs are skipped; the standard plots use aggregate reports only.

The report lists every assessable generation difference, including subgroup-only
signals, and identifies source folders for local patient audits. Loss-only signals
have a separate CSV index. No patient-level IDs are read by the report builder.

## Fixed-cohort selection

The default `fixed` design evaluates 200,000 distinct patients: 80,000 random and
20,000 additional uncommon patients from each of train and tuning. IDs stay fixed
across all contexts, horizons, generation modes and the loss comparison. Short
histories contribute only to eligible cells; they are not replaced.

The rarity scorer never reads BONSAI predictions or weights. A patient-weighted
code-pair frequency reference is fitted on 100,000 uniformly reserved training
patients. Repeated pairs count once per reference patient. A separate 20,000
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
or 1,000,000 candidates per split. Stopping never uses reproduction outcomes.
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

Group-level differences need not produce individual reproduction flags. To
inspect one such cell, including previously unsaved patients, run:

```bash
python -m bonsai.run.audit_memorization_cell \
  --report-dir /path/to/results/cohorts/uncommon \
  --mode code_only --context 32 --horizon 16 --metric overlap \
  --top-per-split 5 --median-per-split 3 \
  --output-dir /path/to/new/cell_audit --device cuda
```

Use the exact combined/cohort/filtered report directory containing the finding.
The command compares all finite scores in that cell, selects examples from both
splits without requiring individual flags, and regenerates only the original
batches containing those IDs. `--subject-id ID ...` requests specific patients
instead. Paths are recovered from the original manifest; use `--checkpoint`,
`--evaluation-cache` or `--vocabulary` overrides if the files moved. Checkpoint
and vocabulary hashes and cache provenance must agree with the original run.

Open `cell_audit/audit.txt` in vim. `summary.txt` reports the full-cell score
distribution and excess, `cell_patient_scores.csv` contains all contributing
IDs, and `selected_patients.csv` records example selection and replay agreement.
New trajectories are labelled as replays, including any disagreement with saved
scores. Matching scalar scores does not prove token-identical original outputs.
High-score/near-median examples explain the comparison; their selection is not
a population estimate or a label that an individual was memorized.

For a rare-code excess, add `--metric rare_codes --all-rare-matches` to the cell
audit command. It selects **every** patient with a saved positive rare-code score
in that cell from both splits, independent of individual flags and horizon. It
also selects zero-score controls (`--median-per-split`, default 3); the top-score
cap does not apply. `rare_code_review_cases.csv` lists all original positive cases.
The audit names each matched, missed or additionally generated rare code, with
true/predicted/matched occurrence counts. `rare_code_matches.csv` provides these
details in tabular form. A generated occurrence matches at most one reference
occurrence; order is ignored, matching the original rare-code metric.

Exact training-population holder counts were not retained in older caches. Those
audits label counts unavailable, rather than estimating them from the evaluated
sample. Newly prepared caches retain the original full-training rare-code counts
and population denominator during the existing frequency scan. Rare-code review
cases are explanatory evidence, not individual memorization diagnoses or changes
to the original run's flag status. Trajectories are checked replays; a replay
disagreement remains visible and is not passed off as an original prediction.
