# Model Registry

The model registry records checkpoint metadata and scientific suitability.

Canonical file:

```text
frozen_checkpoints/model_registry.json
```

## Checkpoint Lifecycle

- `smoke`: tiny debug checkpoints for plumbing tests.
- `candidates`: experimental checkpoints under evaluation.
- `promoted`: approved checkpoints for documented scientific workflows.

Binary checkpoint files are not required to be tracked in git.
Metadata should be tracked.

## Required Registry Fields

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

## Promotion Standard

A model should not be promoted unless it has:

- Reproducible training config.
- Dataset manifest reference.
- Evaluation report.
- Scientific use notes.
- Known limitations.
