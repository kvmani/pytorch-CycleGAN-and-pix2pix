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

## Train CycleGAN

```bash
microi2i train --config configs/train/cyclegan.default.yml
```

## Folder Inference

```bash
microi2i infer --config configs/inference/folder.default.yml
```

## Prepare Paired Microscopy Dataset

```bash
microi2i prepare-dataset --config configs/dataset_prepare/paired_microscopy.default.yml
```

## Evaluate Image Translation

```bash
microi2i evaluate --config configs/evaluation/image_translation.default.yml
```

## Override Config Values

Use dotted-key overrides:

```bash
microi2i train --config configs/train/pix2pix.default.yml --set training.n_epochs=10 --set runtime.gpu_ids=-1
```
