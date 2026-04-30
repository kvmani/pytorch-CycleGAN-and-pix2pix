# Model Registry

The model registry records checkpoint metadata and scientific suitability.

Canonical file:

```text
frozen_checkpoints/model_registry.json
```

## Checkpoint Lifecycle

- `smoke`: tiny debug checkpoints for plumbing tests.
- `candidate`: experimental checkpoints under evaluation.
- `promoted`: approved checkpoints for documented scientific workflows.
- `deprecated`: retained for provenance but no longer recommended.

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
- `model_backend`

## Promotion Standard

A model should not be promoted unless it has:

- Reproducible training config.
- Dataset manifest reference.
- Evaluation report.
- Scientific use notes.
- Known limitations.

## Lifecycle Commands

List the registry:

```bash
microi2i models --details
```

Validate the registry:

```bash
microi2i validate-registry
```

Dry-run a lifecycle update:

```bash
microi2i promote-model \
  --model-id smoke_pix2pix_unet256 \
  --status candidate \
  --metric mae_mean=5.2 \
  --note "Validation report reviewed" \
  --dry-run
```

Compare evaluation reports before promotion:

```bash
microi2i compare-runs --reports run_a/report.json run_b/report.json --metric mae_mean
```

Promotion changes must preserve `lifecycle_history` so future users can reconstruct why a model
was promoted, deprecated, or held as a candidate.
