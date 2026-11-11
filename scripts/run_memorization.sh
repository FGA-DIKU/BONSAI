#!/usr/bin/env bash
# Activate the training environment, edit the settings below, then run:
#   bash scripts/run_memorization.sh

# ---------- Custom settings (use absolute paths) ----------
PYTHON="python"
CHECKPOINT="/path/to/best.ckpt"
RUN_CONFIG="/path/to/original/run/hydra/config.yaml"
DATA_DIR="/path/to/processed/data"
POPULATION="/path/to/training_population.csv"
CUTOFF_DATE="YYYY-MM-DD"                  # Original training cutoff; "" if none.
CONTEXT_LENGTH=8192                      # Original training max_len.

EVALUATION_CACHE="/path/to/cache/memorization_fixed_100k_per_split.pt"
OUTPUT_DIR="/path/to/new/memorization_run" # New or empty folder.
PATIENTS_PER_SPLIT=100000                 # 200,000 total; 80% random, 20% uncommon.
CACHE_PATIENTS=100000                     # Preserve capacity when reusing a cache.
BATCH_SIZE=128
DEVICE="cuda"
SEP_EXCLUDED_FROM_TRAINING_LOSS=true
GENERATE_PLOTS=true                       # Requires the optional 'plots' dependency.
EVERY_STEP_FALLOFF=true                   # CPU-only tables/curves from saved draws.

TRAIN_SPLIT="train"
HELDOUT_SPLIT="tuning"
PROMPTS=(32 256 1024)
CONTINUATIONS=(1 2 4 8 16 32 64)
ZERO_CONTEXT_HORIZONS=(16 32 64 128 256 512)
PUBLIC_START_TOKEN="DOB"                  # Literal public code; no patient value.
TEMPERATURE=1.0
SEED=0

RANDOM_FRACTION=0.8
RARITY_REFERENCE_PATIENTS=100000
RARITY_CALIBRATION_PATIENTS=20000
RARITY_CANDIDATE_LIMIT=1000000
MIN_PATIENTS=50
MIN_UNIQUE_OVERLAP_TOKENS=3
MIN_DIVERSITY_TOKENS=3
DIVERSITY_BASIS="reference"
# ---------- Pipeline (no edits needed below) ----------

set -eo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_DIR"

for file in "$CHECKPOINT" "$RUN_CONFIG" "$POPULATION" \
    "$DATA_DIR/vocabulary.pt" \
    "$DATA_DIR/subject_data_${TRAIN_SPLIT}.pt" \
    "$DATA_DIR/subject_data_${HELDOUT_SPLIT}.pt"; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing input: %s\nEdit the custom settings at the top of this script.\n' "$file" >&2
        exit 1
    fi
done
if [[ "$CUTOFF_DATE" == "YYYY-MM-DD" ]]; then
    printf 'Set CUTOFF_DATE to the original training cutoff (or "" if none).\n' >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]] && [[ ! -d "$OUTPUT_DIR" || -n "$(ls -A -- "$OUTPUT_DIR")" ]]; then
    printf 'Use a new or empty OUTPUT_DIR: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi

sep_args=()
cutoff_args=()
plot_args=()
falloff_plot_args=()
[[ "$SEP_EXCLUDED_FROM_TRAINING_LOSS" == true ]] && sep_args+=(--sep-excluded-from-training-loss)
[[ -n "$CUTOFF_DATE" ]] && cutoff_args+=(--cutoff-date "$CUTOFF_DATE")
[[ "$GENERATE_PLOTS" == true ]] || plot_args+=(--no-plots)
[[ "$GENERATE_PLOTS" == true ]] && falloff_plot_args+=(--plots)

mkdir -p -- "$OUTPUT_DIR" "$(dirname -- "$EVALUATION_CACHE")"
cp -- "$REPO_DIR/scripts/run_memorization.sh" "$OUTPUT_DIR/run_settings.sh"

run_step() {
    local name="$1"
    shift
    printf '\nRunning %s\n' "$name"
    "$PYTHON" -u -m "$@" 2>&1 | tee "$OUTPUT_DIR/$name.log"
}

# 1. Select/cache the shared cohort; run greedy context-conditioned checks.
run_step supplemental bonsai.run.memorization_check \
    --checkpoint "$CHECKPOINT" --run-config "$RUN_CONFIG" \
    --data-dir "$DATA_DIR" --population "$POPULATION" \
    "${cutoff_args[@]}" --context-length "$CONTEXT_LENGTH" \
    --train-split "$TRAIN_SPLIT" --heldout-split "$HELDOUT_SPLIT" \
    --evaluation-cache "$EVALUATION_CACHE" --cache-patients "$CACHE_PATIENTS" \
    --max-patients "$PATIENTS_PER_SPLIT" --sampling-design fixed \
    --random-fraction "$RANDOM_FRACTION" \
    --rarity-reference-patients "$RARITY_REFERENCE_PATIENTS" \
    --rarity-calibration-patients "$RARITY_CALIBRATION_PATIENTS" \
    --rarity-candidate-limit "$RARITY_CANDIDATE_LIMIT" \
    --prompt-events "${PROMPTS[@]}" --continuation-events "${CONTINUATIONS[@]}" \
    "${sep_args[@]}" --export-patient-inspection \
    --min-unique-overlap-tokens "$MIN_UNIQUE_OVERLAP_TOKENS" \
    --min-patients "$MIN_PATIENTS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR/supplemental"

# 2. Primary check: probability sampling from the public token on the same cohort.
run_step context_zero bonsai.run.zero_context_check \
    --report-dir "$OUTPUT_DIR/supplemental" \
    --evaluation-cache "$EVALUATION_CACHE" --max-patients "$PATIENTS_PER_SPLIT" \
    --start-token "$PUBLIC_START_TOKEN" --horizons "${ZERO_CONTEXT_HORIZONS[@]}" \
    --temperature "$TEMPERATURE" --seed "$SEED" \
    --min-unique-tokens "$MIN_UNIQUE_OVERLAP_TOKENS" \
    --min-diversity-tokens "$MIN_DIVERSITY_TOKENS" --diversity-basis "$DIVERSITY_BASIS" \
    --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR/context_zero"

# Optional descriptive analysis; original comparison statuses stay unchanged.
if [[ "$EVERY_STEP_FALLOFF" == true ]]; then
    run_step falloff bonsai.run.zero_context_falloff \
        --report-dir "$OUTPUT_DIR/context_zero" \
        --evaluation-cache "$EVALUATION_CACHE" \
        --output-dir "$OUTPUT_DIR/context_zero/falloff" "${falloff_plot_args[@]}"
fi

# 3. Retain raw supplemental results and add diversity-filtered comparisons.
run_step reanalysis bonsai.run.reanalyze_memorization \
    --report-dir "$OUTPUT_DIR/supplemental" \
    --min-unique-overlap-tokens "$MIN_UNIQUE_OVERLAP_TOKENS" \
    --min-diversity-tokens "$MIN_DIVERSITY_TOKENS" --diversity-basis "$DIVERSITY_BASIS" \
    --output-dir "$OUTPUT_DIR/supplemental_review"

# 4. Combine primary/supplemental, raw/filtered and subgroup reports; plot once.
run_step report bonsai.run.report_memorization \
    --report-dir "$OUTPUT_DIR/supplemental_review" \
    --primary-report "$OUTPUT_DIR/context_zero" \
    --output-dir "$OUTPUT_DIR/report" "${plot_args[@]}"

printf '\nFinished. Start with:\n  %s\n  %s\n' \
    "$OUTPUT_DIR/report/report.txt" "$OUTPUT_DIR/report/flagged_comparisons.csv"
