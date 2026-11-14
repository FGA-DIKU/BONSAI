# Memorization results: reviewer map

Use this guide to find the main results, supporting checks and individual audits
from a completed memorization run. Paths are relative to the run's **output
folder** (`OUTPUT_DIR` in
[`scripts/run_memorization.sh`](scripts/run_memorization.sh)). Optional outputs
depend on which stages were completed.

## 1. Essential paths and review checks

### Open these first

1. **`report/report.txt`** — combined report: primary conclusion, supplemental
   findings, limitations and source-folder paths.
2. **`context_zero/primary_summary.txt`** — primary check, including random,
   uncommon and combined cohorts. Generation starts from a public token with no
   patient history. Check the unfiltered and diversity-filtered results.
3. **`report/flagged_comparisons.csv`** and **`report/flagged_loss_checks.csv`** —
   indexes of group comparisons requiring review, with the source folder for each
   finding. A header-only table has no listed findings; consult the report for
   incomplete or inconclusive checks.

### Supporting evidence

| Check | Saved files to open |
|---|---|
| Scores and differences | Each analysis's `excess.csv`: train/heldout means, differences, confidence intervals and counts. `X` = prompt events; `Y` = continuation events. |
| Loss-based membership check | `supplemental_review/loss_check.csv`; `report/flagged_loss_checks.csv` indexes findings. This is a supplemental check. |
| Patient counts and coverage | Each analysis's `coverage.csv`; `excess.csv` gives metric-specific counts, including rare-code denominators. |
| Repetition and diversity | `patient_flag_rates.csv`, `patient_tails.csv`; compare with `diversity_filtered/` and its `diversity_exclusions.csv`. |
| Figures | `report/plots/primary/` and `report/plots/`; `*_metrics.png` shows means/excess, `cohort_excess.png` compares cohorts, `coverage.png` shows eligibility. Subfolders mirror the analysis views. |

### What to check before accepting the conclusion

- **Status:** `CHECKS PASSED` means required comparisons were assessable and their
  intervals included the reference value. `CHECK` means a difference needs review;
  `INCONCLUSIVE` means required evidence is incomplete.
- **Direction and uncertainty:** `excess` is training minus tuning. An assessable
  interval excluding **0** in either direction needs review; loss AUC uses **0.5**.
  Intervals are pointwise and unadjusted for multiple comparisons.
- **Units and counts:** report differences marked `pp` are percentage points;
  **+2 pp = 0.02** in fractional tables. `n=a/b` means contributing training/tuning
  patients, not the requested cohort size. Check rare-code denominators in particular.
- **Cohorts and filtering:** review random and uncommon subgroups as well as the
  combined result, and compare unfiltered with diversity-filtered results. The
  random cohort is the population-facing check; uncommon and combined samples
  are enriched stress samples.
- **Limits:** a pass or empty finding index does not establish absence of
  memorization. Individual example flags and group-level differences answer
  different questions; an audit contains selected examples, not every contributor.

## 2. Additional reference

Use this section to locate supporting material or decode a report label.

### Full folder map

```text
OUTPUT_DIR/
├── report/                  Combined report, finding indexes and figures
│   └── plots/               primary/ = context zero; other folders = supplemental
├── context_zero/            Primary results, original sampled predictions, audit
│   ├── cohorts/             random/ and uncommon/ subgroup results
│   ├── diversity_filtered/  Separate view excluding low-diversity continuations
│   │   └── cohorts/         Filtered random/ and uncommon/ results
│   └── falloff/             Optional every-step curves and tables
├── supplemental/            Original prompted greedy checks, sampling and audits
├── supplemental_review/     Reanalysis of the same saved supplemental scores
│   ├── cohorts/             random/ and uncommon/ subgroup results
│   └── diversity_filtered/  Filtered results, with their own cohorts/ subfolders
├── run_settings.sh          Copy of the configured pipeline script
└── *.log                    Progress/errors for each pipeline stage
```

`supplemental/` also contains `cohorts/random/` and `cohorts/uncommon/`.
Within each analysis, the root describes the combined sample. The random cohort
is the population-facing check; uncommon and combined cohorts are enriched stress
samples. Filtered results supplement the unfiltered results.

### Sampling, individual audits and provenance

| Check | Saved files to open |
|---|---|
| Analysis summaries | Its `summary.txt`; primary overview: `context_zero/primary_summary.txt`. |
| Sampling and balance | `supplemental/sampling_report.json` and `sampling_balance.csv`; `sampling_patients.csv` lists individual selections. |
| Results by generation length | `context_zero/falloff/report.txt`, `every_step.csv` and optional `falloff_*.png`. These are descriptive, not additional pass/fail checks. |
| Individual examples | Original `context_zero/audit.txt` or `supplemental/audit.txt`. Saved examples are in `flagged_trajectories.jsonl` and `unflagged_examples.jsonl`; individual scores are in `patient_scores.csv`. Reanalysis retains the original trajectory sources. |
| Model, settings and inputs | Each analysis's `run_manifest.json`; root `run_settings.sh` and stage logs. Primary draw provenance: `context_zero/generation_manifest.json` and `generation_assignment.csv`; stored draws: `generations.npy`. |

**Patient-level files contain patient information.**

### What each number in the report means

#### Scores and differences

| Label | Meaning |
|---|---|
| `train` / `tuning` (`unseen`) | Patients used for training versus the heldout comparison group. Tuning may have influenced checkpoint selection. |
| `X` / context | Number of patient events supplied before generation. Primary `X=0` supplies only the public start token, so its model seed length is 1. |
| `Y` / horizon | Number of continuation positions evaluated. Positions include separators (SEP); the public start token is excluded. |
| `exact_match` | Fraction of patients whose entire scored continuation matches in order. In the primary check this includes SEP. |
| `overlap` | Mean of each patient's fraction of true clinical-code occurrences reproduced, ignoring order. Each generated occurrence matches at most one true occurrence; special tokens are excluded. |
| `rare_codes` | Overlap restricted to codes held by fewer than **0.1%** of all filtered training-window patients. Only patients with a true rare-code occurrence in that continuation contribute. This is separate from selection into the uncommon cohort. |
| `excess` | Training mean minus tuning mean. Positive = more reproduction in training patients; negative = more in tuning patients. |
| `95% CI`, `[lower, upper]` | Uncertainty interval around the estimated difference or AUC. For excess, an interval entirely above or below **0** triggers review; for loss AUC, the reference is **0.5**. Intervals apply separately to each comparison and are not adjusted for the number of comparisons. |
| `%` versus `pp` | Percent is a score or rate; **pp** means percentage points between rates. Training 12% versus tuning 10% gives excess **+2 pp**. Aggregate CSVs and supplemental `summary.txt` use fractions: the same excess is **0.02**. Combined `report.txt`, primary summaries and fall-off reports display differences in pp. |
| `loss_train` / `loss_unseen` | Group means of patients' average next-code cross-entropy losses, using the real preceding codes (teacher forcing). Lower means the model assigns higher probability to the observed codes. |
| `AUC` | How well lower patient loss ranks training patients ahead of tuning patients. **0.5** = chance ranking; above 0.5 = training patients tend to have lower loss; below 0.5 = the reverse. Either direction can trigger review when its interval excludes 0.5. AUC is not a fraction of patients memorized. |

For example, `X=32 Y=64 overlap: +2.0 pp [0.5, 3.5]; n=1000/900`
means that after a 32-event prompt, mean clinical-code recall over 64 positions
was 2 percentage points higher for training patients. The interval excludes zero,
so this comparison needs review. The contributing counts were 1,000 training and
900 tuning patients.

#### Counts, coverage and status

| Label | Meaning |
|---|---|
| Requested patients **per split** | Target cohort size for each group separately: 100,000 per split means up to 200,000 patients overall. |
| `n=a/b`, `patients=a/b` | Contributing training/tuning counts, in that order. Counts depend on history length, metric eligibility and diversity filtering; rare-code counts are often smaller. |
| Comparisons assessed / assessable `a/b` | Number of metric comparisons with sufficient patients in both groups and finite intervals, out of those expected/listed by that report. A comparison is one mode × X × Y × metric. |
| Incomplete grid cells | Mode × X × Y combinations whose coverage is incomplete. Each cell can contain several metric comparisons. |
| Structurally inapplicable cells | Combinations excluded by the diversity requirement itself, rather than failing from too few sampled patients. |
| Comparisons needing review | Assessable comparisons whose interval excludes zero, in either direction. This counts comparisons, not distinct patients. |
| `CHECKS PASSED` | All required comparisons were assessable and their intervals included the reference value. |
| `CHECK` | At least one assessable difference requires investigation. It can coexist with incomplete coverage elsewhere. |
| `INCONCLUSIVE` | Required coverage, sampling, settings or estimable intervals were insufficient for an overall pass. |
| Loss `ASSESSABLE` | The loss comparison had adequate data and its AUC interval included 0.5. |

The same patients appear across horizons, modes and views. Counts of comparisons
and patient flags should therefore not be added to estimate people memorized.

#### Diversity, patient examples and sampling

| Label | Meaning |
|---|---|
| Exact match `a/b (p%)` | Patients with an exact continuation match / evaluated patients in that cell, and the corresponding percentage. |
| Low diversity among exact `a/b` | Exact matches classified as low diversity / all exact matches. Total low diversity uses all evaluated patients as its denominator. |
| Minimum distinct codes / diversity threshold | Required number of unique clinical codes. `reference` counts the true continuation; `predicted` counts the generated one. The saved settings identify the basis and threshold. Filtered generation scores use retained patients; supplemental loss/AUC stays unfiltered. |
| Patient flags / reproduction triggers | Examples meeting individual review rules. A trigger count expressed in **cells** counts affected grid cells, not patients. Thresholds are saved in `run_manifest.json`; flags guide audits rather than diagnose memorization. |
| Median / p95 / max | Middle, 95th-percentile and highest patient score in a cell. These describe the score distribution rather than add statistical tests. |
| Random fraction | Share allocated to the random cohort; remaining patients are additional uncommon cases. The combined sample is deliberately enriched. |
| Rarity fitting / calibration counts | Reserved training patients used to estimate code-pair frequencies / set uncommon-selection cutoffs. They are separate from evaluated training patients. |
| Prepared random / uncommon counts | Patients selected into the cache. `uncommon=a/b` reports selected/requested uncommon patients. Cache capacity can exceed the evaluated subset; `complete` records whether selection requirements were met. |

#### Every-step fall-off report

`Y` is the current generated position. **Exact match** requires the entire prefix
through Y to match; **position accuracy** requires only the code at Y to match.
Overlap and rare-code recall use the prefix through Y. `train%` and `tuning%`
are group means; `excess_pp` and `interval_pp` are their difference and interval
in percentage points.

**Fixed** retains patients with references and predictions through the maximum
horizon. **All available** uses everyone eligible at each step, so membership can
change. Rare-code denominators can change in either view as true rare codes first
appear. `adequate_n=false` marks counts below the saved minimum. These curves
describe the saved predictions and leave the original check statuses unchanged.

### Methods

For methods, see
[`docs/memorization_zero_context.md`](docs/memorization_zero_context.md) and
[`docs/memorization_sampling.md`](docs/memorization_sampling.md).

