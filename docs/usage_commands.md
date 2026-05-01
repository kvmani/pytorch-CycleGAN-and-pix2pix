# Usage Commands

These are the canonical commands for the restructured repository.

## List Models

```bash
microi2i models --details
```

With local machine-specific checkpoint overlay:

```bash
microi2i models --overlay frozen_checkpoints/model_registry.local.json --details
```

## Validate Model Registry

```bash
microi2i validate-registry --config configs/registry_validation.default.yml
```

## Train pix2pix

```bash
microi2i train --config configs/train/pix2pix.default.yml
```

Dry-run command and manifest generation without launching training:

```bash
microi2i train --config configs/train/pix2pix.default.yml --dry-run
```

Training runs now emit a stable package before launch:

```text
training_report.json
metrics_log.csv
metrics_log.jsonl
training_summary.html
command.json
run_manifest.json
artifact_manifest.json
```

`training_report.json` contains the resolved legacy command, parsed options, dataset path checks,
dataset manifest warnings, runtime metadata, and launch-blocking errors.

## Train CycleGAN

```bash
microi2i train --config configs/train/cyclegan.default.yml
```

## Create Smoke Data And Run CPU-Safe Smoke Training

Create tiny deterministic pix2pix and CycleGAN datasets:

```bash
microi2i create-smoke-data --config configs/smoke/default.yml
```

Run the smoke gate with dry-run train/infer checks:

```bash
python scripts/smoke_gate.py
```

Run tiny real CPU training only when dependencies are installed and runtime is acceptable:

```bash
python scripts/smoke_gate.py --run-training
```

Dry-run the smoke training commands:

```bash
microi2i train --config configs/train/pix2pix.smoke.yml --dry-run
microi2i train --config configs/train/cyclegan.smoke.yml --dry-run
```

The smoke training configs force CPU execution, one epoch, tiny image sizes, tiny dataset limits,
single-worker loading, and stable checkpoint output under `artifacts/smoke_checkpoints`.
Remove `--dry-run` only when local dependencies are installed and the tiny smoke datasets exist.

Run explicitly on CPU:

```bash
microi2i train --config configs/train/pix2pix.smoke.yml --set runtime.device=cpu
```

Run on a GPU server when CUDA is required:

```bash
microi2i train \
  --config configs/train/pix2pix.default.yml \
  --set runtime.device=cuda \
  --set runtime.gpu_ids=0 \
  --set runtime.require_cuda=true
```

If CUDA is requested but unavailable, MicroI2I fails before launching the legacy training script.

After a real training run, MicroI2I parses legacy `loss_log.txt` into:

```text
metrics_log.csv
metrics_log.jsonl
loss_curves.csv
loss_curves.svg
training_outputs.json
validation_samples.html
validation_samples/
```

## Folder Inference

```bash
microi2i infer --config configs/inference/folder.default.yml
```

Inference runs write `run_manifest.json`, `artifact_manifest.json`, `report.json`, and `report.html` under the configured `output_root`.
Input selection is normalized into `inference_inputs.json` and `inference_inputs.csv` when `inference.inputs` is configured.
When `inference.expected_output_dir` points to generated legacy result images, MicroI2I also packages:

```text
predictions/
batch_summary.json
batch_summary.csv
review.html
comparison_review.html
```

Optional postprocessing can be configured under `inference.postprocess`:

```yaml
postprocess:
  grayscale: true
  resize: [256, 256]
  threshold: null
  auto_contrast: false
  rename_prefix: pred
```

Supported input modes:

```yaml
inference:
  inputs:
    mode: folder      # legacy, single, folder, or manifest
    path: ./images
    recursive: true
    copy_to_run: true
```

Single-image and manifest-driven presets are also available:

```bash
microi2i infer --config configs/inference/single_image.default.yml --dry-run
microi2i infer --config configs/inference/manifest.default.yml --dry-run
```

If `inference.references_dir` points to same-named reference images, `comparison_review.html` shows
prediction/reference pairs for human scientific review.

## Prepare Paired Microscopy Dataset

```bash
microi2i prepare-dataset --config configs/dataset_prepare/paired_microscopy.default.yml
```

Expected source layout:

```text
source/
  specimen_001/pair_001.png
  specimen_002/pair_002.png
```

Output layout:

```text
artifacts/prepared_datasets/paired_microscopy_default/
  train/
  val/
  test/
```

Each output split preserves the relative source path and the run emits `dataset_manifest.json`.

## Prepare Unpaired Microscopy Dataset

```bash
microi2i prepare-dataset --config configs/dataset_prepare/unaligned_microscopy.default.yml
```

## Run Dataset QA

```bash
microi2i data-qa --config configs/dataset_qa.default.yml
```

Dataset QA validates source image folders before training.
It checks for empty roots, unreadable images, duplicate files, shape/channel mismatches, leakage-group assignment, and cross-domain leakage-group overlap.

Outputs:

```text
artifacts/dataset_qa/<dataset_id>/
  dataset_qa_report.json
  dataset_qa_report.html
  contact_sheet.jpg
```

The `microi2i` run folder also receives copies of the QA report and contact sheet for provenance.

## EBSD And Kikuchi Domain Workflows

Domain wrappers expose legacy EBSD/Kikuchi scripts with explicit config parameters and provenance.
They are dry-run by default:

```bash
microi2i run-domain --config configs/domain/ebsd_process_file.default.yml
microi2i run-domain --config configs/domain/ebsd_make_pix2pix.default.yml
microi2i run-domain --config configs/domain/ebsd_make_cyclegan.default.yml
microi2i run-domain --config configs/domain/kikuchi_make_cyclegan.default.yml
```

Override paths before execution:

```bash
microi2i run-domain \
  --config configs/domain/ebsd_make_cyclegan.default.yml \
  --set dry_run=false \
  --set domain.legacy_args.1=C:/data/ebsd/domain_b \
  --set domain.legacy_args.3=artifacts/domain/ebsd/cyclegan
```

Every domain run emits `command.json`, `report.json`, `run_manifest.json`, and `artifact_manifest.json`.

Expected source layout:

```text
domain_a/
  specimen_001/a.png
domain_b/
  specimen_001/b.png
```

Output layout:

```text
artifacts/prepared_datasets/unaligned_microscopy_default/
  trainA/
  trainB/
  valA/
  valB/
  testA/
  testB/
```

## Evaluate Image Translation

```bash
microi2i evaluate --config configs/evaluation/image_translation.default.yml
```

Evaluation matches same-named images under `inputs.predictions_dir` and `inputs.targets_dir`.
It reports fidelity metrics, optional SSIM, and microscopy-oriented structure proxies:

- MAE, RMSE, PSNR, SSIM
- gradient correlation and edge MAE
- histogram L1 distance
- contrast-to-noise proxy delta
- high-frequency energy ratio
- Laplacian sharpness ratio
- EBSD/Kikuchi band contrast delta, band sharpness ratio, and orientation coherence delta

The JSON report also includes `metric_families` groups for fidelity, structure, microscopy, and EBSD/Kikuchi review.
Evaluation also writes `evaluation_outliers.csv` and `evaluation_review.html` when same-named pairs
are available. Configure the outlier ranking:

```yaml
review:
  ranking_metric: mae
  lower_is_better: true
  limit: 5
```

## Compare Runs

Compare two or more evaluation reports by an aggregate metric and optionally write a manual review dashboard:

```bash
microi2i compare-runs --reports run_a/report.json run_b/report.json --metric mae_mean --output artifacts/comparisons/mae.json --html-output artifacts/comparisons/mae.html
```

For metrics where larger is better:

```bash
microi2i compare-runs --reports run_a/report.json run_b/report.json --metric ssim_mean --higher-is-better
```

## Promote Model Lifecycle State

Dry-run a lifecycle update before editing the registry:

```bash
microi2i promote-model \
  --model-id smoke_pix2pix_unet256 \
  --status candidate \
  --metric mae_mean=5.2 \
  --note "Candidate after microscopy validation set review" \
  --reviewer "manual-reviewer-name" \
  --dry-run
```

Remove `--dry-run` only after the checkpoint, dataset manifest, metrics, intended use, limitations,
and documentation have been reviewed.

## Override Config Values

Use dotted-key overrides:

```bash
microi2i train --config configs/train/pix2pix.default.yml --set training.n_epochs=10 --set runtime.gpu_ids=-1
```
