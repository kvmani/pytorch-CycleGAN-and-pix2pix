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

## Train CycleGAN

```bash
microi2i train --config configs/train/cyclegan.default.yml
```

## Folder Inference

```bash
microi2i infer --config configs/inference/folder.default.yml
```

Inference runs write `run_manifest.json`, `artifact_manifest.json`, `report.json`, and `report.html` under the configured `output_root`.

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
It reports MAE, RMSE, PSNR, and SSIM when `scikit-image` is installed.

## Override Config Values

Use dotted-key overrides:

```bash
microi2i train --config configs/train/pix2pix.default.yml --set training.n_epochs=10 --set runtime.gpu_ids=-1
```
