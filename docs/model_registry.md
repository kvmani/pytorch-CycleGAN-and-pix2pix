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

Promotion is a manual scientific judgment. A model should not be promoted until a reviewer has inspected the evidence package, normally including:

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

Use a machine-local overlay for checkpoint paths that should not be committed:

```bash
microi2i models --overlay frozen_checkpoints/model_registry.local.json --details
microi2i validate-registry --overlay frozen_checkpoints/model_registry.local.json
```

Overlay records are keyed by `model_id` and appear under `local_overlay` after merging.
Files matching `frozen_checkpoints/model_registry.local*.json` are ignored by git.

Dry-run a lifecycle update:

```bash
microi2i promote-model \
  --model-id smoke_pix2pix_unet256 \
  --status candidate \
  --metric mae_mean=5.2 \
  --note "Validation report reviewed" \
  --reviewer "manual-reviewer-name" \
  --dry-run
```

Compare evaluation reports before manual promotion review:

```bash
microi2i compare-runs --reports run_a/report.json run_b/report.json --metric mae_mean
```

Promotion changes must preserve `lifecycle_history`, including note and reviewer fields where available, so future users can reconstruct the manual judgment behind why a model was promoted, deprecated, or held as a candidate. The registry records review notes and evidence only; the promotion decision remains manual.
