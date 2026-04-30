# Mission Statement

## Project Mission

Build a scientifically rigorous, extensible, and easy-to-use platform for microscopy image-to-image translation.

The repository starts from CycleGAN and pix2pix, then expands into a research platform for paired and unpaired microscopy workflows including EBSD/Kikuchi refinement, denoising, super-resolution, segmentation-mask generation, and related materials-science image processing tasks.

The repository must also act as a teaching tool.
It should explain the principles, algorithms, mathematics, vocabulary, and scientific assumptions behind the code so students and domain researchers can understand what they are running and why it matters.

## Vision

Researchers and students should be able to:

- Prepare custom paired or unpaired microscopy datasets.
- Train pix2pix and CycleGAN models with reproducible configuration.
- Run inference on single images, folders, and batch datasets.
- Evaluate outputs with image-quality and scientific-task metrics.
- Promote useful checkpoints through a documented model registry.
- Inspect every run through human-readable reports and machine-readable manifests.

## Strategic Scope

- Local CLI-first workflows.
- GPU-compatible training and inference with CPU-safe smoke paths.
- YAML-driven configuration with resolved config snapshots.
- Deterministic dataset preparation and split generation.
- Run, artifact, dataset, and model provenance.
- Documentation that is deep enough for scientific review and accessible enough for new students.
- Sphinx documentation with MathJax equations, professional SVG diagrams, tutorials, and teaching-oriented explanations.

## Success Criteria

- pix2pix and CycleGAN remain trainable and inferable on custom datasets after migration.
- EBSD/Kikuchi workflows have first-class documented commands.
- Every major workflow emits provenance manifests.
- Dataset preparation records split rules, preprocessing, and leakage assumptions.
- Model checkpoints have registry metadata before being treated as research artifacts.
- Documentation explains commands, outputs, assumptions, limitations, and validation metrics.
- The documentation site can be built locally and covers usage, algorithms, architecture, workflows, formulas, and key learning terms.
