# Transfer a model and its token definitions

The bundle contains one explicitly selected checkpoint, its vocabulary, the fitted
numeric/adaptive-code transformations and the YAML definitions needed to recreate
tokens. The receiving site uses its own patients. Patient records, populations,
cohorts, ID mappings, predictions and training state are not packaged.

## Pack, unpack and load

Run from an environment with BONSAI installed:

```bash
python -m bonsai.run.model_bundle pack \
  --checkpoint /path/to/best.ckpt \
  --vocabulary /path/to/vocabulary.pt \
  --data-config /path/to/original/create_data.yaml \
  --numeric-metadata /path/to/numeric_metadata.json \
  --code-mapping /path/to/adaptive_code_mapping.parquet \
  --config /path/to/original/pipeline.yaml \
  --config /path/to/original/event_conversion.yaml \
  --config /path/to/original/premeds.yaml \
  --output model_bundle.zip

python -m bonsai.run.model_bundle unpack model_bundle.zip restored_model
python -m bonsai.run.model_bundle load restored_model --device cpu --attn-type sdpa
```

Use `--no-numeric-metadata` or `--no-code-mapping` only when that transform was
unused. `--config` is repeatable and accepts YAML definitions without embedded
patient records. There is no folder discovery, source snapshot or arbitrary data
attachment. Explicitly supply the original definitions, including custom stages.

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
| `settings/*.yaml` | Explicitly supplied BONSAI and ehr2meds preprocessing definitions. |
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

The standard code-only `BonsaiPretrain` checkpoint works with that branch's
existing external-transfer path. Unpack the ZIP first using this repository's
`model_bundle unpack` command. In the **OPERA environment**, use the extracted
files with OPERA's receiver rather than installing this BONSAI package over
OPERA's fork:

```bash
SOURCE="/absolute/path/to/restored_model"
LOCAL_VOCAB="/absolute/path/to/OPERA/data/vocabulary.pt"

python -m bonsai.run.audit_pretrain_transfer \
  --checkpoint "$SOURCE/model/model.ckpt" \
  --source-vocab "$SOURCE/model/vocabulary.pt" \
  --target-vocab "$LOCAL_VOCAB" --min-token-coverage 0.9

python -m bonsai.run.export_external_encoder \
  --checkpoint "$SOURCE/model/model.ckpt" \
  --source-vocab "$SOURCE/model/vocabulary.pt" \
  --target-vocab "$LOCAL_VOCAB" --minimum-token-coverage 0.9 \
  --seed 42 --output-dir /path/to/new/opera_external_encoder
```

This remaps embeddings by exact token strings, imports the source architecture,
uses SDPA, preserves RoPE layout and handles upstream absolute time in thousands
of hours via OPERA's `scaled_time2vec` representation. Local-only tokens receive
newly initialized embeddings; the actual local vocabulary must pass the coverage
gate. Keep OPERA's local vocabulary separate from the transferred original.
Apply the frozen ehr2meds transforms when creating local data, and preserve the
source training/SEP policy in the supplied YAML definitions.

The export supplies an encoder for OPERA's direct-transfer or local-DAPT paths,
not a fitted OPERA outcome head. Initialize direct and DAPT comparisons from the
same remapped export, as described in OPERA's `docs/external_bonsai_transfer_v1.md`.

Checked with synthetic weights/data against `opera/leukemia`: reordered token
IDs transferred correctly, source/receiver hidden states matched within `4.8e-7`
on CPU, the direct export strict-loaded, and a one-step DAPT checkpoint passed
`audit_dapt_to_opera`. Low token coverage was correctly rejected. This does not
validate the real server vocabulary, clinical preprocessing or model performance.
Current `BonsaiValuePretrain` transfers fail because OPERA uses different numeric
embedding/head tensor names; those require a receiver-side adapter before use.
