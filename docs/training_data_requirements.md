# Training Data Requirements

## Supported Dataset Types

### Paired pix2pix

Use paired input/target images where each sample has a known correspondence.
The initial accepted layout is inherited from pix2pix:

```text
dataset/
  train/
  val/
  test/
```

Each paired image may be stored as concatenated `{A,B}` images or as a future manifest-backed pair list.

### Unpaired CycleGAN

Use separate domains:

```text
dataset/
  trainA/
  trainB/
  testA/
  testB/
```

Domain meaning must be documented in `dataset_manifest.json`.

### Single-Folder Inference

Use for applying a trained generator to one domain:

```text
input_images/
  image_001.png
  image_002.png
```

The inference config must state the expected source domain.

## Microscopy Metadata

Dataset manifests should capture when available:

- Material or specimen family.
- Imaging modality.
- Acquisition conditions.
- Pixel size or scale calibration.
- Preprocessing and cropping policy.
- Split leakage group.

## Leakage Control

Validation and test splits must avoid leakage across:

- Same specimen.
- Same field of view.
- Same scan series.
- Same augmentation family.
- Same experiment batch, when known.

## File Formats

Initial workflows should support common image formats through PIL/OpenCV.
TIFF and scientific metadata support should be added deliberately with documented assumptions.
