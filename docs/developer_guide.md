# Developer Guide

This guide describes how to add production-quality functionality to MicroI2I.

## Development Principles

- Put reusable behavior under `src/microi2i/`.
- Keep scripts thin and limited to command entry points.
- Use typed config contracts for public workflows.
- Emit provenance artifacts for every workflow.
- Add tests and docs in the same change as behavior.

## Adding A Workflow

1. Add or extend a config contract in `src/microi2i/core/`.
2. Implement workflow logic in the relevant package module.
3. Add a CLI command or subcommand in `microi2i.app.cli`.
4. Emit `run_manifest.json`, `artifact_manifest.json`, and `report.json`.
5. Add unit tests for pure logic.
6. Add integration tests for CLI behavior and artifacts.
7. Add Sphinx documentation with exact commands and output examples.

## Adding A Metric

1. Implement the metric as a pure function.
2. Add synthetic tests with known expected values.
3. Document the formula with MathJax-compatible LaTeX.
4. Explain interpretation, limitations, and microscopy relevance.
5. Add the metric to evaluation reports.

## Adding A Model Backend

Each backend must define:

- model family name,
- train entry point,
- inference entry point,
- required config fields,
- checkpoint metadata,
- expected dataset layout,
- validation metrics,
- known failure modes.

Backends should be compared against simpler baselines when possible.
GAN-based improvements must be validated scientifically, not only visually.

## Before Handoff

Run:

```bash
python scripts/check_repo.py
```

If the command fails, fix the issue or document the blocker before handoff.
