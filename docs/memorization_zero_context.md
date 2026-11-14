# Context zero on the same patients

The primary context-zero comparison uses the **same fixed patient IDs and
random/uncommon labels** as the supplemental context grid. There is no independent
pool of generated sequences and no search against other patients in this test.
Each eligible patient receives one independently sampled continuation, assigned
before scoring. Exact match, occurrence-limited overlap, rare-code reproduction,
per-patient tails, diversity counts and train-minus-tuning excess use the existing
functions. Random, uncommon and combined results are reported separately.

Only the fixed public `DOB` code is supplied to the model, with code embeddings
and transformer positional encoding. Patient context is zero; actual seed length
is one. No real DOB value, age, time, segment, numeric value, future features or
SEP positions enter generation. Predictions are sampled from the full softmax at
temperature 1, appended autoregressively, and reused across shorter horizons.

Reference continuations start immediately after each record's DOB anchor. The
anchor is located explicitly, not assumed to be the first code. Missing/duplicate
anchors stop the run; patients are never replaced to fit the test. Reference
history before DOB is audit evidence only. Every horizon requires that many true
positions after DOB. Short histories contribute only where eligible, as in the
supplemental checks, and decode only to their longest eligible horizon.

## Use one cohort for both runs

To run the complete pipeline, edit the custom settings at the top of
[`scripts/run_memorization.sh`](../scripts/run_memorization.sh), then run:

```bash
bash scripts/run_memorization.sh
```

The script defaults to **100,000 patients per split**. It runs supplemental
checks, the primary context-zero check, diversity reanalysis and the combined
report in sequence, stopping on errors. Each step has a log in `OUTPUT_DIR`,
which also retains a copy of the configured script. Standard figures are generated
at the end, with optional every-step curves in the fall-off stage; set
`GENERATE_PLOTS=false` for text/CSV reports only.

For **100,000 patients total**, use 50,000 per split: 40,000 random plus 10,000
uncommon in training and the same allocation in tuning. For 200,000 total, use
100,000 per split. The size flag is always **per split**.

Build a new 50,000-per-split cache and run the supplemental checks:

```bash
python -u -m bonsai.run.memorization_check \
  --checkpoint /path/to/best.ckpt \
  --run-config /path/to/run/hydra/config.yaml \
  --data-dir /path/to/processed/data \
  --evaluation-cache /local/path/memorization_50k_per_split.pt \
  --output-dir /path/to/new/supplemental_100k_total \
  --max-patients 50000 --cache-patients 50000 \
  --batch-size 128 --sep-excluded-from-training-loss \
  --export-patient-inspection --device cuda

python -u -m bonsai.run.zero_context_check \
  --report-dir /path/to/new/supplemental_100k_total \
  --output-dir /path/to/new/context_zero_100k_total \
  --max-patients 50000 --start-token DOB \
  --horizons 16 32 64 128 256 512 \
  --min-diversity-tokens 3 --diversity-basis reference \
  --batch-size 128 --device cuda --plots
```

Omit the SEP flag if SEP targets were included in training loss; preserve original
population/cutoff/context overrides. Context zero freely samples SEP either way.
There is **no `--generations` setting**: the count follows the selected patients.
The existing default remains 100,000 per split unless overridden as above.
Each analysis automatically writes its own text report and audit. `--plots`
requests immediate primary figures; the combined report is a separate final
command below and generates both primary and supplemental figures by default.

Alternatively prepare the same cache with `--prepare-sampling-only --device cpu`
and run context zero first. If reusing a larger prepared cache, keep its original
`--cache-patients` capacity for supplemental cache-signature compatibility and use
the same smaller `--max-patients` in both checks. Fixed nested subsets are shared.
The cohort-ID fingerprint is recorded; combined reports reject different patient
memberships instead of silently presenting them as the same cohort.

## Outputs and post-hoc review

Outputs retain the existing conventions: `excess.csv`, `coverage.csv`,
`patient_scores.csv`, `patient_tails.csv`, `patient_flag_rates.csv`, `summary.txt`,
`report.txt`, `audit.txt`, and separate `cohorts/random/` and `cohorts/uncommon/`.
Loss/AUC remains a supplemental shared check, not a separate context-zero loss
attack. A pass requires all applicable, adequately sampled excess intervals to
include zero; missing rare-code denominators are inconclusive. Pointwise intervals
are not equivalence bounds or proof of absence of memorization.
All primary subgroup findings are listed in the primary overview. The combined
`flagged_comparisons.csv` includes primary and supplemental findings with explicit
source-folder paths, even when a combined-cohort estimate does not flag.

`generation_assignment.csv` maps stored matrix rows to the exact patient IDs,
cohort and reference offset. `generations.npy` stores assigned predictions.
`generation_manifest.json` records hashes, per-patient generated lengths and RNG
settings. Unused positions are `-1`, never scored as tokens. Patient information
and generated sequences remain on the data server.
The overlap-flag diversity threshold (`--min-unique-tokens`, default 3) does not
remove raw scores. `--min-diversity-tokens 3` additionally creates a separate
filtered comparison. It preserves unfiltered results, writes every exclusion
denominator, and includes filtered primary cohort plots when `--plots` is used.
Supplemental filtered comparisons use the existing `reanalyze_memorization`
command after inference; no new model predictions are needed.

Change diversity settings without model inference:

```bash
python -m bonsai.run.zero_context_check \
  --report-dir /path/to/new/supplemental_100k_total \
  --reuse-generations /path/to/new/context_zero_100k_total \
  --output-dir /path/to/new/context_zero_review \
  --max-patients 50000 --min-unique-tokens 3 \
  --min-diversity-tokens 3 --diversity-basis reference --device cpu

python -m bonsai.run.zero_context_check \
  --audit-only /path/to/new/context_zero_100k_total \
  --subject-id PATIENT_ID --output-dir /path/to/new/patient_audit

python -m bonsai.run.audit_memorization_cell \
  --report-dir /path/to/new/context_zero_100k_total/cohorts/random \
  --mode code_only --context 0 --horizon 16 --metric rare_codes \
  --all-rare-matches --output-dir /path/to/new/rare_audit
```

Reuse retains the original assignments and RNG draws. Longer horizons need new
inference if the stored continuations do not cover them. Raw and filtered reports
are both retained. Audits use original stored predictions and reference-only
history, not model replays. Every positive rare-code patient in the selected cell
can be recovered independently of the individual exact/overlap trigger.

Build a combined report with primary and supplemental results:

```bash
python -m bonsai.run.report_memorization \
  --report-dir /path/to/new/supplemental_100k_total \
  --primary-report /path/to/new/context_zero_100k_total \
  --output-dir /path/to/new/combined_report
```

## Every-step fall-off from existing predictions

No GPU inference is needed. Using the completed **unfiltered combined**
context-zero report, run:

```bash
python -m bonsai.run.zero_context_falloff \
  --report-dir /path/to/context_zero_results \
  --output-dir /path/to/context_zero_results/falloff --plots
```

Omit `--plots` for table/text only. `every_step.csv` and `report.txt` report all
saved steps for exact-prefix match, occurrence-limited overlap, rare-code recall
and accuracy at the individual position. They include train/tuning means,
train-minus-tuning excess, pointwise intervals and metric-specific denominators.
Three optional figures cover combined/random/uncommon cohorts. The combined
report links to these outputs when they are saved in `context_zero_results/falloff`.
The run script enables this stage with `EVERY_STEP_FALLOFF=true`.

The `fixed` view retains only patients with references and saved draws through
the maximum horizon. The `all_available` view uses shorter draws too; its patient
membership can change with the step. Rare-code denominators can change even in
the fixed view as the first true rare code appears. All views are unfiltered;
these descriptive curves add no pass/fail tests and do not alter existing gates.
The default maximum is the stored matrix width; `--max-horizon 128`, for example,
defines a fixed cohort eligible through 128, provided those predictions exist.
Stored `-1` padding is never scored. Missing steps are not regenerated.

## Runtime and sensitivity

Patient-dependent generation/scoring is approximately linear in eligible patient
counts at a fixed grid, model, GPU and batch size. Halving 200,000 total patients to
100,000 roughly halves this work. Full-population loading, rarity fitting and some
cache setup do not halve. Generation to a longer horizon is more expensive than
just its token count: this implementation recomputes the growing context. Actual
cost also depends on how many records support each horizon.

The observed 20,000-total supplemental run took 40 minutes at batch size 128.
At similar throughput, 100,000 total is roughly 3 hours 20 minutes for that grid;
200,000 total is roughly 6 hours 40 minutes. The context-zero cell is additional:
budget roughly 2–4 hours per 100,000 patients if all decode to 512, often less with
short-history grouping. These are planning estimates, not measured GPU results.

One sampled continuation per patient has limited exact-match sensitivity: it may
reproduce another patient's trajectory and score zero against its assigned target.
Increasing patient numbers reduces sampling uncertainty;
it does not make a finite negative diagnostic a proof that no record is memorized.
