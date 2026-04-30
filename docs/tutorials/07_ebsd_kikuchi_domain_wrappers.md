# EBSD And Kikuchi Domain Wrappers

Legacy EBSD and Kikuchi scripts remain available, but they must now be launched through explicit
MicroI2I configs when used as part of reproducible workflows.

## Dry-Run First

```bash
microi2i run-domain --config configs/domain/ebsd_process_file.default.yml
microi2i run-domain --config configs/domain/ebsd_make_pix2pix.default.yml
microi2i run-domain --config configs/domain/ebsd_make_cyclegan.default.yml
microi2i run-domain --config configs/domain/kikuchi_make_cyclegan.default.yml
```

These configs default to `dry_run: true` because the legacy scripts may otherwise rely on
machine-specific paths or large local datasets.

## Execute With Explicit Paths

```bash
microi2i run-domain \
  --config configs/domain/ebsd_process_file.default.yml \
  --set dry_run=false \
  --set domain.legacy_args.1=C:/data/ebsd/raw/specimen_001.ang \
  --set domain.legacy_args.3=artifacts/domain/ebsd/specimen_001
```

## Required Review

Before using domain outputs for model training:

- run `microi2i data-qa`,
- inspect the generated report and contact sheet,
- preserve domain-specific acquisition notes in dataset metadata,
- document thresholds, masks, cropping, and filtering assumptions.
