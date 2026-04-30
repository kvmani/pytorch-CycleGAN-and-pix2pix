# AGENTS.md - Repository Working Contract

This file defines how developers and automation agents must work in this repository.
The repository is being restructured from a general CycleGAN/pix2pix codebase into a scientific microscopy image-to-image translation platform.

If a future task conflicts with this document, use this priority order:

1. Scientific correctness, traceability, and reproducibility
2. Preservation of current useful functionality after migration
3. Clear user-facing workflows and documentation
4. Maintainable package architecture
5. Speed of delivery

## Mission Alignment

All changes must align with `docs/mission_statement.md`.
The repository must evolve toward:

- Paired and unpaired microscopy image-to-image translation
- pix2pix and CycleGAN training/inference on custom datasets
- EBSD/Kikuchi denoising, refinement, and related scientific workflows
- Extension to super-resolution, noise cancellation, segmentation-mask generation, and restoration tasks
- Reproducible experiment artifacts with machine-readable metadata
- Beginner-friendly command-line workflows for students and collaborators

## Current Restructure Rule

Backward-compatible file paths are not mandatory during this branch-based restructure.
Preserving functionality is mandatory.

Legacy scripts and modules may be moved, renamed, or absorbed only after equivalent behavior exists in the new package and is documented with tests or smoke checks.

## Architecture Rules

- Prefer package modules under `src/microi2i/` over monolithic top-level scripts.
- Keep scripts thin; orchestration belongs in library modules.
- Keep data preparation, training, inference, evaluation, manifests, and reporting separate.
- Keep model backends as CLI/package execution adapters; do not introduce web services for model execution.
- Use config files and command overrides instead of hardcoded machine paths.
- Do not hardcode checkpoint paths, dataset roots, host names, or scientific constants without documenting them.
- Public APIs must use type hints and clear NumPy- or Google-style docstrings.
- Avoid import-time side effects in library modules.
- New package code must be testable without requiring GPU hardware, internet access, or large datasets.
- Heavy model execution belongs behind explicit smoke/integration gates, not import-time or default unit tests.
- Every new model backend must include config presets, docs, tests, registry metadata expectations, failure modes, and smoke behavior when feasible.

## Scientific Provenance

Every major workflow must write machine-readable metadata:

- `run_manifest.json`
- `artifact_manifest.json`
- `report.json`
- `dataset_manifest.json` where datasets are created or transformed

Metadata must capture command, resolved config, code state, timestamp, environment, seed, dataset references, model references, and generated artifacts where feasible.

## Documentation Sync

Any behavior change must update relevant documentation in the same change.
At minimum:

- `README.md` for user-facing command changes
- `docs/usage_commands.md` for canonical commands
- `docs/code_provenance_and_manifests.md` for output schema changes
- `tests/README.md` for validation protocol changes

Markdown links inside the repository must be repository-relative.

Every behavior change must include tests, docs, and provenance updates in the same change unless the change is explicitly documentation-only or test-only.

## Sphinx Teaching Documentation Standard

This repository must function as both a research codebase and a teaching tool.
All substantial features must be documented in the Sphinx documentation site under `docs/`.

Mandatory documentation expectations:

- Keep `docs/index.md` as the canonical documentation landing page.
- Build documentation with `python scripts/build_docs.py --html-only` for routine checks.
- Use MyST Markdown with MathJax-compatible LaTeX for equations.
- Include mathematical formulations for losses, metrics, preprocessing transforms, and scientific calculations when relevant.
- Explain key ML and microscopy terms for students and interdisciplinary collaborators.
- Add or update SVG diagrams under `docs/diagrams/` for architecture, model structure, workflows, and data flow when behavior is non-trivial.
- SVG diagrams must use professional visual styling: readable typography, rounded boxes, consistent spacing, gradients where helpful, and a coherent color palette.
- Avoid relying on inline Mermaid as final publication documentation; use committed SVG diagrams for stable HTML/PDF output.
- Documentation must cover philosophy and scientific assumptions, not only command syntax.
- Any new model family must include architecture explanation, original citation, local modifications, assumptions, failure modes, and validation guidance.
- Any new metric must include its mathematical formula, symbol definitions, interpretation, and limitations.

## Testing Expectations

- Unit tests are mandatory for pure logic, config schemas, metrics, registry validation, and manifest generation.
- Integration tests are mandatory for CLI workflows, generated artifacts, dataset preparation, and report creation.
- CPU-only smoke tests are mandatory for train/infer plumbing before legacy workflows are removed.
- Regression tests are mandatory before migrating EBSD/Kikuchi behavior.
- Scientific metric changes require synthetic-data tests with known expected values and Sphinx documentation with formulas.
- Dataset splitting and leakage controls require deterministic tests.
- Every substantial documentation change must keep `python scripts/build_docs.py --html-only` green.
- The local quality gate is `python scripts/check_repo.py`; it must pass before handoff or commit unless a blocker is documented.

## Production Quality Gates

Before merge, changes must satisfy:

- `python -m pytest tests`
- `python scripts/microi2i_cli.py validate-registry`
- `python scripts/build_docs.py --html-only`
- `python scripts/check_repo.py`

Feature work must also satisfy the relevant gate:

- New workflow: unit tests, CLI integration test, manifest/report validation, docs page, and usage command.
- New metric: formula docs, unit tests, interpretation notes, and failure/limitation notes.
- New model backend: interface compliance test, config preset, registry metadata, architecture docs, and smoke path.
- Legacy migration: parity test, old/new artifact comparison where practical, and removal notes.

## What To Avoid

- Silent scientific fallbacks.
- Unversioned output schemas.
- Undocumented preprocessing defaults.
- Moving current workflows without an equivalent documented command.
- Mixing local experiment data with source-controlled package structure.
