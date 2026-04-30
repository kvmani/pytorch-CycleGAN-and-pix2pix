# Repository Blueprint

This is the target structure for the branch-based restructure.

```text
pytorch-CycleGAN-and-pix2pix/
  AGENTS.md
  README.md
  pyproject.toml
  configs/
    train/
    inference/
    dataset_prepare/
    evaluation/
    registry_validation.default.yml
  docs/
    adr/
    diagrams/
    tutorials/
  examples/
  frozen_checkpoints/
    model_registry.json
    smoke/
    candidates/
    promoted/
  src/
    microi2i/
      app/
      core/
      dataops/
      domain/
      evaluation/
      inference/
      io/
      manifests/
      models/
      pipelines/
      training/
      utils/
  scripts/
  tests/
    unit/
    integration/
    smoke/
```

## Migration Policy

Backward-compatible paths are not required.
Functionality preservation is required.

Existing top-level code such as `train.py`, `test.py`, `data/`, `models/`, `options/`, `util/`, `EBSD/`, and `kikuchi/` can be absorbed into `src/microi2i/` once equivalent commands and tests exist.

## Canonical Interfaces

The new command surface is:

```bash
microi2i train --config configs/train/pix2pix.default.yml
microi2i train --config configs/train/cyclegan.default.yml
microi2i infer --config configs/inference/folder.default.yml
microi2i prepare-dataset --config configs/dataset_prepare/paired_microscopy.default.yml
microi2i evaluate --config configs/evaluation/image_translation.default.yml
microi2i models --details
microi2i validate-registry
```

## First Implementation Layer

The first layer establishes:

- Documentation contracts.
- Config parsing.
- Manifest generation.
- Registry validation.
- CLI orchestration around current working scripts.

Later layers migrate internals into package modules and remove the legacy layout.
