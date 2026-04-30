# Usage Commands

These are the canonical commands for the restructured repository.

## List Models

```bash
microi2i models --details
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

## Folder Inference

```bash
microi2i infer --config configs/inference/folder.default.yml
```

Inference runs write `run_manifest.json`, `artifact_manifest.json`, `report.json`, and `report.html` under the configured `output_root`.
When `inference.expected_output_dir` points to generated legacy result images, MicroI2I also packages:

```text
predictions/
batch_summary.json
batch_summary.csv
review.html
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

## Compare Runs

Compare two or more evaluation reports by an aggregate metric:

```bash
microi2i compare-runs --reports run_a/report.json run_b/report.json --metric mae_mean
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
  --dry-run
```

Remove `--dry-run` only after the checkpoint, dataset manifest, metrics, intended use, limitations,
and documentation have been reviewed.

## Override Config Values

Use dotted-key overrides:

```bash
microi2i train --config configs/train/pix2pix.default.yml --set training.n_epochs=10 --set runtime.gpu_ids=-1
```
