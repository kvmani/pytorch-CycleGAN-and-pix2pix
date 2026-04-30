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
- Use config files and command overrides instead of hardcoded machine paths.
- Do not hardcode checkpoint paths, dataset roots, host names, or scientific constants without documenting them.
- Public APIs must use type hints and clear docstrings.
- Avoid import-time side effects in library modules.

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

## Testing Expectations

- Add unit tests for config parsing, manifest generation, and registry validation.
- Add integration or smoke tests before replacing legacy train/infer behavior.
- CPU-only smoke tests must remain available.
- Scientific metric changes require tests or documented validation examples.

## What To Avoid

- Silent scientific fallbacks.
- Unversioned output schemas.
- Undocumented preprocessing defaults.
- Moving current workflows without an equivalent documented command.
- Mixing local experiment data with source-controlled package structure.
