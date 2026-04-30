# Documentation Principles

This repository treats documentation as part of the scientific product.

## Required Documentation Rules

1. Every workflow must have an exact runnable command.
2. Every generated output must be documented with path, purpose, and schema version.
3. Every scientific metric must describe what it measures and what it does not prove.
4. Every preprocessing step must document assumptions and parameter defaults.
5. Every model family must cite the original method and identify local modifications.
6. Every behavior change must update documentation in the same change.
7. Diagrams should live as static assets under `docs/diagrams/` when they become publication or onboarding material.
8. Markdown links must be repository-relative.

## Documentation Layers

- Mission and scope.
- Quick-start usage.
- Dataset preparation.
- Training and inference.
- Scientific validation.
- Provenance and manifest schemas.
- Model registry and checkpoint lifecycle.
- Developer architecture.
- Student learning path and glossary.

## Command Documentation Standard

Every command example should identify:

- Required inputs.
- Output directory.
- Config file.
- Important overrides.
- Expected manifests and reports.
- CPU/GPU assumptions.
