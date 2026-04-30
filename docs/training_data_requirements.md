# Training Data Requirements

## Supported Dataset Types

### Paired pix2pix

Use paired input/target images where each sample has a known correspondence.
The initial accepted source layout may be nested by specimen or acquisition group:

```text
source/
  specimen_001/pair_001.png
  specimen_001/pair_002.png
  specimen_002/pair_003.png
```

The phase-2 materializer copies images into pix2pix aligned layout:

```text
prepared_dataset/
  train/
  val/
  test/
```

Each paired image is currently expected to already be in pix2pix concatenated `{A,B}` form.
Future phases may add separate input/target pair-list materialization.

### Unpaired CycleGAN

Use separate domains:

```text
domain_a/
  specimen_001/a_001.png
domain_b/
  specimen_001/b_001.png
```

The materializer writes CycleGAN layout:

```text
prepared_dataset/
  trainA/
  trainB/
  valA/
  valB/
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

The current split implementation groups samples by parent folder by default.
Set `leakage_group_policy.regex` to extract a specimen or scan ID from file paths when folder grouping is insufficient.

## Deterministic Preprocessing

`microi2i prepare-dataset` can apply deterministic preprocessing while materializing the dataset.
Supported policy keys are:

- `color_mode`: `preserve`, `rgb`, or `grayscale`
- `center_crop`: `[width, height]`
- `random_crop`: `[width, height]`, seeded by the split policy and relative path
- `resize`: `[width, height]`
- `pad_to`: `[width, height]`
- `letterbox`: `[width, height]`
- `fill`: integer pad value

Example:

```yaml
preprocessing:
  color_mode: rgb
  center_crop: [512, 512]
  resize: [256, 256]
  letterbox: null
  fill: 0
```

Every copied sample receives a deterministic `global_id` in `dataset_manifest.json`.

## Dataset QA Before Training

Run dataset QA before training:

```bash
microi2i data-qa --config configs/dataset_qa.default.yml
```

QA checks include:

- empty source roots,
- unreadable images,
- duplicate image files,
- width/height/channel mismatches,
- missing leakage-group IDs,
- cross-domain leakage-group overlap for unpaired datasets.

The report should be reviewed before launching long training jobs.

## File Formats

Initial workflows should support common image formats through PIL/OpenCV.
TIFF and scientific metadata support should be added deliberately with documented assumptions.
