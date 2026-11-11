# Transfer a model and its token definitions

The bundle contains one explicitly selected checkpoint, its vocabulary, the fitted
numeric/adaptive-code transformations, and optional YAML definitions for
recreating tokens. The receiving site uses its own patients. Patient records, populations,
cohorts, ID mappings, predictions and training state are not packaged.

## Intended workflow

Use the receiving site's patients with the **source model's representation**:

1. Recreate MEDS events with matching ehr2meds definitions and the frozen mappings,
   normalization limits and lab-bin boundaries.
2. Tokenize with the bundled vocabulary, preserving every token ID. Lab bins are
   ordinary code tokens such as `LAB//BIN_2`; no value-extension model is needed.
3. Use the pretrained encoder for local fine-tuning, probing or OPERA training;
   alternatively, continue pretraining on the local adaptation population first,
   then use that adapted encoder downstream.

**Do not fit a destination vocabulary, refit lab bins or add local token rows.**
Unknown codes follow the source code-mapping/filtering policy and the frozen
vocabulary's `[UNK]` fallback. OPERA can wrap the weights in its own checkpoint
format, but the vocabulary and embedding rows stay unchanged.

This path passed synthetic preprocessing and CPU model checks described below.
The bundle supplies assets, not an automatic receiving-site pipeline: local raw
source tables/paths, effective YAML settings, installed stage resources and
model-loading configuration still need to be connected. No real server bundle
or receiving-site patient data was available for these checks.

## Pack, unpack and load

Run from an environment with BONSAI installed:

```bash
python -m bonsai.run.model_bundle pack \
  --checkpoint /path/to/best.ckpt \
  --vocabulary /path/to/vocabulary.pt \
  --ehr2meds-dir /path/to/completed/ehr2meds_run \
  --output model_bundle.zip

python -m bonsai.run.model_bundle unpack model_bundle.zip restored_model
python -m bonsai.run.model_bundle load restored_model --device cpu --attn-type sdpa
```

`--ehr2meds-dir` searches the selected run recursively for exactly one
`numeric_metadata.json`/`.parquet` and one
`adaptive_code_mapping.json`/`.parquet`. It exports only their allowed transform
columns, never the directory's patient shards. Use the output directory of the
original preprocessing run, not the ehr2meds source-code checkout.

If either artifact is absent or several candidates exist, packing fails rather
than selecting a fit or assuming it was unused. Explicit `--numeric-metadata`
and `--code-mapping` paths override discovery for their respective roles; they
also support custom filenames. Use `--no-code-mapping` or
`--no-numeric-metadata` only for a transform that was unused. The model and
vocabulary remain explicitly selected: the vocabulary is normally produced by
BONSAI data creation, rather than ehr2meds.

**Configs are optional attachments.** Add `--data-config create_data.yaml` and/or
repeat `--config pipeline.yaml` for any YAML definitions you want to include.
Model constructor settings already travel in the cleaned checkpoint and
manifest; loading weights does not require YAML. Recreating the same inputs
still requires the correct preprocessing definitions/settings, either supplied
in the bundle or already available at the receiving site. Configs are not
automatically collected from the directory.

Install the matching BONSAI/ehr2meds versions and dependencies at the destination,
including ehr2meds' reference resources. Record those versions alongside the
transfer. The utility does not install or manage environments or source code.

The pipeline YAML alone may reference defaults rather than contain all effective
settings. Include the original event/preMEDS definitions and any overridden
numeric, lab-text or unknown-code settings with `--config`. In particular retain
`numeric_value_columns`, `numeric_value_column_groups`, the bin-code separator,
lab-text `value_map`/namespace/separator, and unknown-token filters. Resolve
Hydra defaults/environment-dependent transformation settings before handing
them over; input/output paths can be adapted locally. The bundle copies YAML
definitions, not the files referenced by paths inside them.

If preMEDS enrichment is used, the matching ehr2meds checkout must provide
`resources/shak_to_region.parquet`, `resources/c_spec_to_specialty.parquet` and
`resources/sor2_contact_mapping.parquet` as required by the selected definitions.
These are static reference tables. They are not in the model ZIP; do not assume
a wheel installation includes them at the paths used by preMEDS. Source patient
tables, contact joins, subject-ID mappings and split/shard files stay local.

## Contents

| File | Purpose |
| --- | --- |
| `model/model.ckpt` | Model tensors and scalar constructor settings; optimizer/callback/training state removed. |
| `model/vocabulary.pt` | Original token-to-ID mapping, including `[SEP]`. |
| `preprocessing/numeric_metadata.*` | Per-code normalization limits, bin edges/representatives and optional hard bounds; raw values, counts and identifiers removed. |
| `preprocessing/code_mapping.*` | `code` → `adaptive/mapped_code` only; counts/profiles removed. |
| `settings/*.yaml` | Optional, explicitly supplied BONSAI and ehr2meds definitions. |
| `manifest.json` | File paths, model class and constructor settings. |

The fitted export supports `aggregate_numeric_metadata` →
`annotate_numeric_values` → optional `join_numeric_bins`, with the standard
per-code schema. Custom column names/grouping keys (`code_modifiers`) require
a corresponding exporter change. The older `bin_numeric_values_fast` path uses
different metadata/bin structs and optional `custom_bins_filepath`; it is not
covered by this numeric exporter. Its actual fitted bins and custom YAML would
need a separate explicit export, not a refit on destination data.
Selection/projection does not
certify arbitrary token strings, weights or configuration contents as anonymous.
ZIP's built-in CRC checks detect archive corruption; extraction rejects unsafe
paths, symbolic links and unlisted files. Existing archives/folders are never
overwritten. This simpler format is version 2; repack older bundles with the
explicit-file command above.

## Use the model and recreate tokens

```python
from pathlib import Path
from bonsai.functional.model_loading import load_bundle

bundle = load_bundle(Path("restored_model"), device="cpu", attn_type="sdpa")
model = bundle.model
vocabulary = bundle.vocabulary
numeric_metadata = bundle.numeric_metadata  # Path or None if unused
code_mapping = bundle.code_mapping          # Path or None if unused
```

The loader supports standard/value pretraining and fine-tuning networks, loads
weights strictly and requires the vocabulary to define every embedding-table ID.
It uses
evaluation mode. Older checkpoints omit attention dropout; zero is recorded for
inference, where dropout is disabled. Backend overrides may introduce numerical
differences. Ordinary PyTorch model saving remains separate from bundle creation;
there is no loaded-model re-bundling method or training-resumption support.

At the receiving site:

1. Adapt input paths and source tables to local data. Preserve event-code semantics,
   units, text normalization, numeric column groups and preprocessing order.
2. Remove `fit_adaptive_code_mapping` and `aggregate_numeric_metadata` from the
   destination pipeline. Point `mapping_filepath` on **both**
   `apply_adaptive_code_mapping` and `finalize_adaptive_code_metadata` at the
   bundled mapping. Point `numeric_metadata_filepath` on `annotate_numeric_values`
   at the bundled numeric metadata. Use fresh outputs; locally fitted metadata
   takes precedence.
3. Preserve numeric bounds and binning: bin indices count edges less than or equal
   to the normalized value. Keep optional `join_numeric_bins` and its separator
   (`//BIN_` by default) consistent with the original vocabulary.
4. Set BONSAI's `tokenizer.vocabulary` to the extracted vocabulary path. Preserve
   separator insertion, code cutoffs/exclusions, numeric inputs and feature
   construction. Never refit the vocabulary or fitted mappings on local data.

## Receiving the bundle in OPERA (`opera/leukemia`)

Use the standard `BonsaiPretrain` model with joined lab-bin code tokens. Unpack
with this repository's command, then use OPERA's environment and receiver. Avoid
installing this BONSAI package over OPERA's fork.

First create local MEDS data using the frozen transforms described above. In
OPERA's data-creation YAML, set an absolute path:

```yaml
tokenizer:
  vocabulary: /absolute/path/to/restored_model/model/vocabulary.pt
  # Preserve source cutoffs and separator settings as well.
```

Leave the vocabulary frozen throughout data creation, DAPT and downstream use.
The processed output's `vocabulary.pt` must have the **same dictionary** as the
bundled one; serialization hashes can differ even for equal dictionaries. Check
this after data creation:

```python
from pathlib import Path
import torch

source = Path("/absolute/path/to/restored_model/model/vocabulary.pt")
processed = Path("/absolute/path/to/local_processed/vocabulary.pt")
assert torch.load(source, weights_only=True) == torch.load(processed, weights_only=True)
```

OPERA's existing external-encoder utility can import the source architecture and
wrap the weights in its native format. Give it the **same vocabulary as both
source and target**, with a 100% coverage requirement. Its general-purpose
remapping then becomes an identity mapping: every embedding row is copied to the
same ID, with no new token embeddings. This is format/architecture adaptation,
not a change to the data representation.

```bash
SOURCE="/absolute/path/to/restored_model"
SHARED_VOCAB="$SOURCE/model/vocabulary.pt"
SOURCE_ROPE_LAYOUT="interleaved"  # Confirm against the source training code.
ENCODER="/path/to/new/opera_source_encoder"

python -m bonsai.run.audit_pretrain_transfer \
  --checkpoint "$SOURCE/model/model.ckpt" \
  --source-vocab "$SHARED_VOCAB" --target-vocab "$SHARED_VOCAB" \
  --min-token-coverage 1.0 \
  --override "paths.initialization_rope_layout=$SOURCE_ROPE_LAYOUT" \
  --output-json /path/to/new/transfer_audit.json

python -m bonsai.run.export_external_encoder \
  --checkpoint "$SOURCE/model/model.ckpt" \
  --source-vocab "$SHARED_VOCAB" --target-vocab "$SHARED_VOCAB" \
  --minimum-token-coverage 1.0 --source-rope-layout "$SOURCE_ROPE_LAYOUT" \
  --seed 42 --output-dir "$ENCODER"
```

Require `local_non_special_token_coverage=1.0` and `new_local_tokens=0` in the audit.
Use `ENCODER/encoder.ckpt` for the direct downstream arm. This transfers the
pretrained encoder; local outcome heads are still trained on local outcomes.
Probing can freeze the encoder, while fine-tuning can update it as usual.

For the adaptation arm, initialize from that same encoder, use the same source
vocabulary and the newly processed local patient records, and start a fresh
optimizer. For example, in OPERA's environment:

```bash
LOCAL_DATA="/absolute/path/to/local_processed"
DAPT_RUN="/path/to/new/local_dapt"

python -m bonsai.run.pretrain \
  --config-name pretrain_external_bonsai_dapt_v1 \
  "paths.dir=$LOCAL_DATA" "paths.vocab=$SHARED_VOCAB" \
  "paths.population=$LOCAL_DATA/cohorts/population_full.csv" \
  "paths.initialization_ckpt=$ENCODER/encoder.ckpt" \
  "paths.initialization_vocab=$ENCODER/vocabulary.pt" \
  "paths.initialization_min_token_coverage=1.0" \
  "hydra.run.dir=$DAPT_RUN"

python -m bonsai.run.audit_dapt_to_opera \
  --checkpoint "$DAPT_RUN/version_0/best.ckpt" --vocabulary "$SHARED_VOCAB"
```

Adapt the population path to the actual local output layout. Review the DAPT
budget, splits, target/SEP policy and context length against the intended
experiment; this example inherits OPERA's existing DAPT training defaults.
The source encoder architecture is imported explicitly. Use the adapted
checkpoint for downstream training with the same vocabulary and feature
construction. This is continued pretraining, not resumption of the source
optimizer or training run.

### Feature compatibility

OPERA's receiver imports source width, layers, head count and RoPE layout, and
uses SDPA. It represents source absolute time (thousands of hours) with
`scaled_time2vec` applied to OPERA's absolute-time inputs in hours. Preserving
token IDs alone is insufficient: age, time origin/units, segment construction,
code cutoffs, SEP insertion and event ordering must also agree. Import the source
architecture instead of keeping an unrelated local Fourier/value-model config.
Binned lab tokens do not require a numeric embedding or regression head.

For upstream BONSAI, SDPA uses `interleaved`; FlashAttention also uses
`interleaved` from commit `a3fa6ac` (2026-09-29). Earlier upstream FlashAttention
weights use `non_interleaved`. Establish this from the code that trained the
checkpoint, rather than the code present when it is packed. The original flat
hyperparameters do not record the layout. OPERA's export records the declaration
for subsequent loads; the source repository's own loader does not thereby gain
support for older non-interleaved weights.

### What was verified

On 2026-11-11, using synthetic data and local OPERA commit `3b90bc7`:

- All 12 source bundle tests passed with the ehr2meds checkout available. The new
  integration check exports frozen transforms, applies the actual ehr2meds
  mapping/annotation/bin-joining stages to local lab values, and tokenizes with
  the bundled vocabulary. It checks values below/at/above source bounds, exact
  bin boundaries, unchanged IDs, unknown-code fallback and unchanged predictions.
- With the **identical source vocabulary**, a 256-wide, two-layer, four-head
  encoder transferred with 100% coverage, zero new tokens and bit-identical code
  embedding rows. Source/OPERA hidden states agreed within `4.8e-7` on CPU.
- OPERA's tokenizer preserved the bundled lab-bin IDs and mapped unknown local
  codes to `[UNK]` without growing the vocabulary.
- The exported encoder strict-loaded. One local DAPT optimizer step produced a
  checkpoint accepted by `audit_dapt_to_opera`.
- The same-vocabulary encoder worked with OPERA contrastive embeddings and
  multi-outcome predictions on a padded batch, using newly initialized heads.
- Earlier receiver checks also passed 5 transfer tests and 15 RoPE tests
  (2 skipped). Those included different-vocabulary cases; those cases are not
  the intended workflow here.

The preprocessing test exercises stage functions, not a complete raw-table
extraction/sharding run. With the current ehr2meds checkout, stage registration
loads `configs/MEDS/*.yaml` relative to the working directory: run its pipeline
from that checkout with the package installed so stage entry points are available.
The configured static reference resources must also be present.

The tests used PyTorch 2.14.1; the frozen-preprocessing test used MEDS 0.4.1,
MEDS-transforms 0.6.7 and Polars 1.44.2. They establish compatibility for this
synthetic path, not clinical preprocessing correctness on unseen source tables,
clinical performance, full training convergence or GPU FlashAttention behavior.
Run the saved-artifact audits in the actual receiving environment.

### What these checks cannot guarantee

This is not a foolproof end-to-end transfer. Discovery validates names and fitted
schemas, not that a selected checkpoint, vocabulary and transforms came from the
same training run. Configs are copied rather than executed or fully resolved.
The receiver must still check event definitions, clinical units, unknown-code
handling and feature construction against the source. Successful loading and
identity vocabulary transfer do not validate those semantics or clinical
performance. Keep the actual-artifact audit and a small local preprocessing/model
smoke test before full training.
