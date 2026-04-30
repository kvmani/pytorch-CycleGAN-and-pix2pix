# Development Roadmap

This roadmap moves MicroI2I beyond feature parity with the original CycleGAN/pix2pix repository into a production-grade scientific platform for microscopy image-to-image translation.

## Phase 1: Governance And Quality Gates

- Enforce repository contracts in `AGENTS.md`.
- Treat tests, documentation, and provenance as mandatory parts of every behavior change.
- Maintain `scripts/check_repo.py` as the local quality gate.
- Require `python -m pytest tests`, registry validation, and Sphinx HTML builds before merge.

## Phase 2: Dataset QA And Scientific DataOps

- Add and expand `microi2i data-qa`.
- Validate paired and unpaired layouts before training.
- Detect empty folders, unreadable images, dimension/channel mismatches, duplicates, and split leakage.
- Generate `dataset_qa_report.json`, `dataset_qa_report.html`, and visual contact sheets.
- Support microscopy metadata fields: modality, material, magnification, pixel size, acquisition notes, specimen ID, and operator.
- Add deterministic split manifests with per-sample global IDs.
- Add configurable preprocessing policies: resize, center crop, random crop, pad/letterbox, grayscale/RGB conversion.

## Phase 3: Training Workflow Upgrade

- Validate dataset manifests and model configs before launch.
- Save resolved configs, commands, environment, git state, metrics logs, and reports in training run folders.
- Add debug/smoke mode for tiny CPU-safe runs.
- Track fixed validation samples per epoch.
- Emit structured JSON/CSV loss logs.
- Write interruption-safe failure reports.
- Produce HTML training summaries with loss curves and generated image panels.
- Add presets for pix2pix microscopy, CycleGAN microscopy, EBSD/Kikuchi denoising, super-resolution, and noise cancellation.

Current status: smoke dataset generation, smoke-aware command construction, structured legacy loss
parsing, and validation sample packaging are implemented. Remaining work is live loss streaming,
loss-curve plots, and fully executable CPU smoke training in CI.

## Phase 4: Inference And Batch Review

- Support single image, folder, recursive folder, and manifest-driven inference.
- Normalize outputs into one run package with predictions, panels, reports, and manifests.
- Add batch summary CSV/JSON with per-image metadata.
- Add postprocessing hooks: grayscale conversion, resize, threshold, contrast normalization, and file renaming.
- Add human-review panels for scientific inspection.
- Preserve partial outputs and failure reports for interrupted runs.

Current status: folder, single-image, recursive folder, and manifest input normalization now emit
`inference_inputs.json`/`.csv` before legacy inference. Same-named references can be reviewed in
`comparison_review.html`. Remaining work is direct legacy dataroot staging for non-dry runs and
metric-aware best/worst review panels.

## Phase 5: Scientific Evaluation And Metrics

- Expand metrics beyond MAE/RMSE/PSNR/SSIM.
- Add edge preservation, gradient correlation, histogram distance, noise reduction, contrast-to-noise ratio, high-frequency recovery, edge sharpness, and resolution consistency.
- Add EBSD/Kikuchi proxies: band contrast, sharpness, and downstream indexing hooks when available.
- Add per-sample and aggregate reports, best/worst sample panels, and run comparison.
- Document every metric with formula, interpretation, and limitations.

## Phase 6: Model Registry And Lifecycle

- Extend model lifecycle states: `smoke`, `candidate`, `promoted`, and `deprecated`.
- Add `microi2i promote-model` and `microi2i compare-runs`.
- Track dataset manifest, metrics, intended use, limitations, and promotion decision evidence.
- Support a git-ignored local overlay registry for machine-specific checkpoint paths.
- Require objective promotion criteria and documentation before scientific use.

## Phase 7: New Model And Algorithm Frontiers

- Add a common model-backend interface for train, infer, evaluate, and metadata export.
- Preserve pix2pix and CycleGAN as first backends.
- Add future adapters for CUT, Pix2PixHD, ESRGAN/Real-ESRGAN-style super-resolution, diffusion/one-step translation, and non-GAN restoration baselines.
- Each backend requires config presets, docs, tests, registry support, and failure-mode notes.

Current status: the CLI-only model execution adapter layer has begun with `legacy_pix2pix` and
`legacy_cyclegan`. The next extensions should add native adapter tests, backend registry metadata,
and then new research adapters such as CUT or restoration baselines.

## Phase 8: Teaching-Grade Documentation

- Expand Sphinx docs into a full teaching site.
- Cover GAN philosophy, scientific risks, pix2pix, CycleGAN, hyperparameters, metrics, EBSD/Kikuchi workflows, super-resolution, denoising, model registry, and backend development.
- Use SVG diagrams for architecture/workflows and MathJax LaTeX for formulas.
- Link every new doc from `docs/index.md`.

## Phase 9: Automation And CI Readiness

- Keep `scripts/check_repo.py` as the local gate.
- Add optional lint/type checks once the migrated code stabilizes.
- Add CI when ready: install dependencies, run tests, validate registry, build docs, and run CLI smoke commands.
- Keep generated artifacts ignored and source artifacts reproducible.

Current status: `scripts/smoke_gate.py` provides a dry-run smoke gate and an opt-in real CPU training
mode. The local `scripts/check_repo.py --include-smoke` path runs this smoke gate after tests,
registry validation, and docs.

## Definition Of Done

A phase is complete only when:

- behavior is implemented in package modules,
- tests cover unit, integration, smoke, or regression risk as appropriate,
- docs explain usage, principles, assumptions, outputs, and limitations,
- provenance artifacts are emitted,
- and the local quality gate passes.
