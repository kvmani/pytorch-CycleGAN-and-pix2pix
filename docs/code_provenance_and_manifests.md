# Code Provenance And Manifest Standards

All major workflows must emit metadata that can be inspected by humans and parsed by scripts.

## Run Manifest

File: `run_manifest.json`

Required fields:

- `schema_version`
- `run_id`
- `workflow`
- `started_utc`
- `finished_utc`
- `status`
- `command`
- `config_path`
- `resolved_config`
- `code_state`
- `environment`
- `artifacts`

The workflow closeout report is written as `report.json` unless the workflow already wrote a domain-specific `report.json`.
In that case, closeout writes `run_report.json` and leaves the workflow report intact.

## Artifact Manifest

File: `artifact_manifest.json`

Required fields:

- `schema_version`
- `run_id`
- `created_utc`
- `files`

Each file entry should include:

- `path`
- `kind`
- `description`
- `exists`
- `size_bytes`
- `sha256` when practical

## Dataset Manifest

File: `dataset_manifest.json`

Required fields:

- `schema_version`
- `dataset_id`
- `source_roots`
- `task_type`
- `layout`
- `output_dataset_dir`
- `split_policy`
- `preprocessing`
- `sample_counts`
- `leakage_group_policy`
- `copied_files`

The initial supported layouts are `pix2pix_aligned` and `cyclegan_unaligned`.

## Dataset QA Report

File: `dataset_qa_report.json`

Required fields:

- `schema_version`
- `dataset_id`
- `task_type`
- `status`
- `metadata`
- `source_roots`
- `root_summaries`
- `summary`
- `leakage_group_policy`
- `samples`
- `issues`
- `contact_sheet`

The matching `dataset_qa_report.html` and `contact_sheet.jpg` support human review before training.

## Training Report

File: `training_report.json`

Required fields:

- `schema_version`
- `status`
- `preflight`

The preflight section records parsed legacy options, model family, dataset mode, experiment name,
script and dataset path checks, runtime settings, warnings, and errors. Training packages must also
include `metrics_log.csv`, `metrics_log.jsonl`, and `training_summary.html` even when legacy internals
are still responsible for the actual optimization loop.

## Training Outputs

File: `training_outputs.json`

Required fields:

- `schema_version`
- `experiment_dir`
- `loss_log`
- `loss_rows`
- `validation_sample_count`
- `validation_samples`
- `validation_samples_html`

When legacy training writes `loss_log.txt`, MicroI2I parses it into `metrics_log.csv` and
`metrics_log.jsonl`. When generated sample images exist under the legacy `web/images` directory,
MicroI2I copies a bounded panel into `validation_samples/` and writes `validation_samples.html`.

## Smoke Dataset Manifest

File: `smoke_dataset_manifest.json`

Smoke datasets are deterministic tiny datasets intended for CPU-safe plumbing checks.
The manifest records output roots, image size, sample count, seed, dataset layouts, and generated samples.

## Domain Workflow Report

File: `report.json`

EBSD/Kikuchi domain wrappers use `microi2i.domain_workflow_report.v1` and record the domain,
task, explicit parameters, resolved legacy command, and script path. Domain configs must default to
`dry_run: true` unless they operate only on generated smoke data.

## Evaluation Report

File: `report.json`

Evaluation reports use `microi2i.evaluation_report.v1` and must include per-sample metrics plus
aggregate metrics. Current aggregate fields include pixel fidelity, SSIM when available, edge/gradient
metrics, histogram distance, contrast-to-noise proxy delta, high-frequency energy ratio, and Laplacian
sharpness ratio.

## Model Registry

File: `frozen_checkpoints/model_registry.json`

Each model entry should include:

- `model_id`
- `display_name`
- `model_family`
- `task_type`
- `framework`
- `checkpoint_path_hint`
- `input_assumptions`
- `training_dataset`
- `metrics`
- `scientific_use`
- `limitations`
- `status`
- optional `lifecycle_history`

Allowed statuses are `smoke`, `candidate`, `promoted`, and `deprecated`.
Promotion to `candidate` or `promoted` must be backed by a dataset manifest, evaluation report,
documented limitations, and reviewable provenance.

## Schema Versions

Use `microi2i.<artifact>.v1` for initial schemas.
Schema changes must update this document and tests.
