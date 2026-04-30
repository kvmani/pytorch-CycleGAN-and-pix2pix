# Prepare A Paired Microscopy Dataset

This tutorial prepares a pix2pix-style paired microscopy dataset.

## Source Layout

```text
source/
  specimen_001/pair_001.png
  specimen_002/pair_002.png
```

Each image should already contain the concatenated pix2pix `{A,B}` pair.

## Command

```bash
microi2i prepare-dataset --config configs/dataset_prepare/paired_microscopy.default.yml \
  --set source_roots=[D:/microscopy/paired_source] \
  --set output_dataset_dir=D:/microscopy/prepared_pix2pix
```

## Outputs

```text
prepared_pix2pix/
  train/
  val/
  test/
```

The run directory also contains `dataset_manifest.json`, `run_manifest.json`, and `artifact_manifest.json`.
