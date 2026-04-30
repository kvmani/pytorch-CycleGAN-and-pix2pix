# Development Roadmap

## Phase 1: Foundation

- Create `AGENTS.md` and foundational docs.
- Add package scaffold under `src/microi2i`.
- Add config parsing, manifest writing, and registry validation.
- Add canonical CLI commands that wrap current behavior.

## Phase 2: Dataset Operations

- Implement deterministic dataset preparation.
- Emit `dataset_manifest.json`.
- Add leakage-aware split policies.
- Document paired, unpaired, EBSD, and Kikuchi dataset recipes.

## Phase 3: Training And Inference Orchestration

- Move orchestration into package modules.
- Save resolved configs and manifests for train/infer runs.
- Generate HTML/JSON review reports.
- Preserve pix2pix and CycleGAN behavior through new commands.

## Phase 4: Scientific Evaluation

- Add paired image metrics.
- Add denoising and super-resolution metrics.
- Add EBSD/Kikuchi quality proxies.
- Produce per-sample and aggregate reports.

## Phase 5: Model Lifecycle

- Establish smoke, candidate, and promoted checkpoint lifecycle.
- Validate model registry metadata.
- Add promotion criteria and scientific limitation notes.

## Phase 6: Full Migration

- Move legacy modules into `src/microi2i`.
- Retire old top-level scripts after new commands are tested and documented.
- Keep the repository package-first and documentation-first.
