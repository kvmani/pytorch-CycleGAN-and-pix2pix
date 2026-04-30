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
9. The Sphinx documentation site is the canonical documentation product.
10. Teaching content must explain principles, algorithms, mathematical formulations, and key terms, not only command usage.
11. Equations must use MathJax-compatible LaTeX where mathematical clarity matters.
12. Workflow, code architecture, and model architecture pages should include professional SVG diagrams with coherent color schemes, rounded boxes, gradients, and readable typography.
13. New metrics must include formulas, symbol definitions, interpretation, and limitations.
14. New model families must include architecture diagrams or schematics, citations, assumptions, and failure modes.

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
- Algorithm and metric formulations.
- Professional SVG diagrams for architecture, model structure, and workflows.

## Command Documentation Standard

Every command example should identify:

- Required inputs.
- Output directory.
- Config file.
- Important overrides.
- Expected manifests and reports.
- CPU/GPU assumptions.

## Sphinx Build Standard

Routine HTML build:

```bash
python scripts/build_docs.py --html-only
```

Full offline review build, when Playwright is available:

```bash
python scripts/build_docs.py
```

The docs should remain useful as:

- a user manual,
- a developer guide,
- a scientific audit trail,
- and a teaching text for GAN-based microscopy image translation.
