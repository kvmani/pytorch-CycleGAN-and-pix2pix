# Documentation Principles

This repository treats documentation as part of the scientific product.
Documentation is also part of the testable production surface: substantial code changes must update the Sphinx site and keep the HTML build green.

## Required Documentation Rules

1. Every workflow must have an exact runnable command.
2. Every generated output must be documented with path, purpose, and schema version.
3. Every scientific metric must describe what it measures and what it does not prove.
4. Every preprocessing step must document assumptions and parameter defaults.
5. Every model family must cite the original method and identify local modifications.
6. Every behavior change must update documentation in the same change.
7. Diagrams should live as static assets under `docs/diagrams/` when they become publication or onboarding material; final Sphinx documentation should prefer committed SVG over Mermaid.
8. Markdown links must be repository-relative.
9. The Sphinx documentation site is the canonical documentation product.
10. Teaching content must explain principles, algorithms, mathematical formulations, and key terms, not only command usage.
11. Equations must use MathJax-compatible LaTeX where mathematical clarity matters.
12. Workflow, code architecture, and model architecture pages should include PyTex-standard professional SVG diagrams with coherent blue/purple/green/orange palettes, rounded boxes, gradients, shadows where useful, consistent spacing, and readable Arial/Helvetica typography.
13. New metrics must include formulas, symbol definitions, interpretation, and limitations.
14. New model families must include architecture diagrams or schematics, citations, assumptions, and failure modes.
15. New workflows must document expected inputs, exact commands, generated artifacts, and failure modes.
16. New dataset formats must document layout, metadata fields, leakage risks, and QA checks.
17. Documentation-only changes that alter navigation must update `docs/index.md`.

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


## SVG Visual Standard

Use the PyTex project as the visual reference for stable diagrams: light blue gradient backgrounds, dark navy headings, rounded white cards, restrained shadows, semantic color accents, and clearly labeled arrows. Diagrams should be source-controlled SVG files, not screenshots. Use Mermaid only for quick design sketches; convert accepted architecture and workflow diagrams into committed SVG assets before treating the documentation as complete.

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

## Documentation Gate

The following command must pass before handoff for any substantial docs or behavior change:

```bash
python scripts/build_docs.py --html-only
```

Known warnings from inherited legacy upstream docs may be tolerated temporarily, but new pages should not introduce avoidable heading, link, or toctree warnings.
