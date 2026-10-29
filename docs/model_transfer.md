# Transfer a model and its token definitions

The destination site uses **its own patients** with the transferred model,
vocabulary and frozen ehr2meds transformations. Source-site patient records,
cohorts, subject ID mappings, populations, outcomes, predictions and training
splits have no role in this export.

## What is exported

| Component | Exported content | Needed to reproduce |
| --- | --- | --- |
| Model | Each selected run's `.ckpt`, rewritten with only `model.*` weight tensors, scalar model hyperparameters and Lightning version; `hparams.yaml` and nearest parent Hydra `config.yaml` / `overrides.yaml` | Matching network/class, causal/value options, sequence length, prediction head and attention settings. Full run config matters: checkpoint hyperparameters omit the class and `attention_dropout`. Optimizer, callback, metric and other training state are discarded. |
| Vocabulary | Validated `vocabulary.pt`: token string → integer ID only | Exact token IDs, including `[SEP]`; never rebuild this vocabulary on destination data. |
| Numeric transformations | `code`, `numeric/bin_1`, `numeric/bin_99`, `numeric/bin_edges`, `numeric/bin_representatives`, optional `numeric/hard_minimum` / `numeric/hard_maximum` | Frozen normalization, clipping, bin assignment and representative values. Raw training-value lists, identifiers, counts and other columns are discarded. |
| Adaptive truncation | `code` and `adaptive/mapped_code` only | Original source-code → model-code mapping. Counts, profiles, reasons and other columns are discarded. No training data or refitting is needed to apply it. |
| Implementations/settings | Package `.py`/`.sh` files, YAML config definitions, installation metadata, guide and checksum manifest | Feature construction, event namespaces, text-value normalization, numeric column groups, stage order, tokenizer exclusions/cutoffs and bin-code representation. |
| Reference resources | Only the known `shak_to_region.parquet`, `c_spec_to_specialty.parquet`, `sor2_contact_mapping.parquet` reference files, when present in repository resources | Original geographic/specialty enrichment definitions. Other resource files are not automatically copied. |

The export supports the current standard **per-code** ehr2meds transformation
schema. Required columns and unique code keys are checked. Custom column names
or modifier-dependent transformations need an explicit exporter extension;
do not rename or drop their grouping keys to force them into this schema.

The snapshots must match the model's original preprocessing implementation and
settings. Supply preprocessing definitions without embedded patient records.
File selection and schema projection do not certify arbitrary YAML, source code,
token strings or model weights as anonymous; these remain model/configuration
assets supplied by the exporter.

## Pack and unpack

Packing requires BONSAI's torch and Polars dependencies to validate/export assets;
unpacking uses only Python's standard library. Run from the BONSAI checkout root:

```bash
python -m bonsai.run.model_bundle pack \
  --ehr2meds-dir /path/to/ehr2meds/checkout \
  --ehr2meds-output-dir /path/to/original/MEDS/output \
  --model-dir /path/to/selected/model/run \
  --data-dir /path/to/BONSAI/processed/dataset \
  --config /path/to/original/pipeline.yaml \
  --config /path/to/original/event_conversion_config.yaml \
  --config /path/to/original/premeds_config.yaml \
  --output model_bundle.zip

python -m bonsai.run.model_bundle unpack model_bundle.zip restored_model
```

- `--ehr2meds-dir` is the checkout containing `pyproject.toml`.
- `--ehr2meds-output-dir` is searched for a unique `numeric_metadata.json`
  and `adaptive_code_mapping.parquet`. Only their transformation columns are
  exported; the surrounding data and intermediate shards are never included.
  Explicit `--numeric-metadata` / `--code-mapping` paths support custom locations
  and JSON/Parquet inputs. Declare `--no-numeric-metadata` or `--no-code-mapping`
  only if that transform was unused.
- `--data-dir` supplies `vocabulary.pt` and `logs/hydra/config.yaml`, not subject
  data. Ensure this saved config belongs to the original data creation run.
  `--vocabulary` and `--data-config` override these paths. Without `--data-dir`,
  supply the config explicitly; vocabulary can be discovered in `--model-dir`.
- `--config` accepts additional YAML preprocessing definitions only. There is
  no unrestricted file-attachment option. Config trees do not identify which
  historical settings/overrides were used: include the actual definitions and
  preserve relevant environment values without copying patient-specific inputs.

Checkpoints are read with `weights_only=True`; unsupported pickled objects cause
an error instead of falling back to unrestricted loading. Originals are not
modified. `unzip -l model_bundle.zip` lists the payload. The manifest records
exported-file checksums, repository commit/dirty status and transformation paths.

## Recreate tokens at the destination site

1. Install the extracted `source/BONSAI` and `source/ehr2meds` in compatible
   Python 3.12 environments. Match the original dependency versions, numeric
   precision and attention backend as appropriate; installation declarations
   are included, installed environments and dependencies are not vendored.
2. Adapt paths and source-table definitions to the destination site's own data.
   Retain the model's event-code semantics, units, text synonyms, unknown-code
   filtering and preprocessing order. Patient join tables and subject identities
   are entirely local to that site; no source-site mappings are transferred.
3. Remove `fit_adaptive_code_mapping` and `aggregate_numeric_metadata` from the
   destination pipeline. Point `mapping_filepath` on **both**
   `apply_adaptive_code_mapping` and `finalize_adaptive_code_metadata` at the
   extracted `preprocessing/code_mapping.*`. Point `numeric_metadata_filepath`
   on `annotate_numeric_values` at `preprocessing/numeric_metadata.*`.
   Use fresh output directories: local fitted metadata takes precedence.
4. Keep numeric column groups, plausibility bounds and binning behavior unchanged.
   Bin indices count edges **less than or equal to** the normalized value.
   Preserve optional `join_numeric_bins`, its source column and separator
   (`//BIN_` by default), so generated code strings match the vocabulary.
5. Use the original BONSAI data settings and set `tokenizer.vocabulary` to the
   absolute extracted `model/vocabulary.pt` path. Preserve separator insertion,
   code cutoffs/exclusions, numeric input column and feature construction.
   Instantiate the matching network and load the exported weights for inference.

A synthetic, non-patient example can verify token/bin behavior across sites.
The bundle supplies model/token assets, not a standalone prediction service.
Source-site cohort recreation and training reproduction are outside its scope.
Extraction verifies checksums, rejects unsafe paths/symlinks and requires a new
destination. Python callers can use `pack_bundle` and `unpack_bundle` from
`bonsai.functional.model_bundle`.
