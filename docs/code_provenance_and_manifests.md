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
- `split_policy`
- `preprocessing`
- `sample_counts`
- `leakage_group_policy`
- `created_utc`

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

## Schema Versions

Use `microi2i.<artifact>.v1` for initial schemas.
Schema changes must update this document and tests.
