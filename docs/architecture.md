# Code Architecture

MicroI2I separates scientific workflow concerns into small modules so runs are reproducible and inspectable.

```{image} diagrams/code_architecture.svg
:alt: MicroI2I code architecture
:class: architecture-diagram
```

## Module Roles

- `microi2i.app`: command-line interface and workflow orchestration.
- `microi2i.io`: YAML loading and dotted-key override handling.
- `microi2i.core`: typed contracts for workflow configuration.
- `microi2i.dataops`: deterministic dataset preparation and split materialization.
- `microi2i.training`: training command construction and future training orchestration.
- `microi2i.inference`: inference command construction and output packaging.
- `microi2i.evaluation`: scientific image-quality metrics.
- `microi2i.manifests`: run, artifact, dataset, and report metadata.
- `microi2i.plugins`: model registry and extension metadata.

## Design Principle

Scripts should remain thin.
Scientific behavior should live in importable modules with tests.
Every major workflow must emit enough metadata for another researcher to understand how the result was produced.
