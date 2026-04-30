# Test Strategy

Tests protect scientific behavior and migration safety.

## Required Test Areas

- Config loading and dotted-key overrides.
- Manifest and report schema generation.
- Model registry validation.
- Dataset split determinism.
- CPU-only smoke train/infer paths.
- EBSD/Kikuchi workflow preservation before legacy scripts are removed.

## Current Migration Gate

The first restructuring phase must at least verify:

- `microi2i` package imports.
- Config files load.
- Manifest helpers write valid JSON.
- Registry validation catches malformed entries.
- CLI help works.
