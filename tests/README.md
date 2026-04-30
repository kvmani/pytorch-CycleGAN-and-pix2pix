# Test Strategy

Tests protect scientific behavior and migration safety.

The repository is intended to become a production-quality research platform.
Tests are not optional paperwork; they are part of the scientific provenance of the codebase.

## Required Test Areas

- Config loading and dotted-key overrides.
- Manifest and report schema generation.
- Model registry validation.
- Dataset split determinism.
- CPU-only smoke train/infer paths.
- EBSD/Kikuchi workflow preservation before legacy scripts are removed.
- Dataset QA, leakage detection, and per-sample global IDs.
- Scientific metrics on synthetic arrays with known expected values.
- CLI integration behavior and generated artifact structure.
- Sphinx documentation build for substantial documentation changes.

## Test Layers

### Unit Tests

Unit tests cover pure functions and contracts:

- config parsing and validation
- split assignment and leakage grouping
- metric calculations
- manifest/report schema payloads
- model registry validation
- command construction

Unit tests must not require GPU, network, large files, or external services.

### Integration Tests

Integration tests cover behavior across modules:

- CLI dry-runs
- dataset preparation using tiny synthetic datasets
- report and manifest generation
- inference packaging from small folders
- registry validation through the CLI

### Smoke Tests

Smoke tests prove that train/infer plumbing can run in CPU-safe debug mode.
They should use tiny datasets and minimal epochs/iterations.

### Regression Tests

Regression tests are required before moving or replacing legacy EBSD/Kikuchi scripts.
They should capture command construction, expected output locations, and representative artifact behavior.

### Scientific Validation Tests

Scientific metrics must be tested with synthetic fixtures where the expected value is known.
Examples:

- identical images produce zero MAE/RMSE and infinite or maximal PSNR behavior
- constant offset images produce predictable MAE/RMSE
- leakage grouping keeps related samples in one split

## Required Local Gates

Run before handoff or commit:

```bash
python -m pytest tests
python scripts/microi2i_cli.py validate-registry
python scripts/build_docs.py --html-only
python scripts/check_repo.py
```

`scripts/check_repo.py` is the single command quality gate and should remain aligned with this document.

Optional smoke workflow gate:

```bash
python scripts/check_repo.py --include-smoke
```

Real CPU training smoke runs remain opt-in:

```bash
python scripts/smoke_gate.py --run-training
```

## Acceptance Rule

A feature is not complete until:

- tests cover the behavior,
- docs describe the workflow and outputs,
- manifests/reports record provenance,
- and the local quality gate passes.
