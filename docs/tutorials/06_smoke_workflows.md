# Smoke Workflows

Smoke workflows are tiny deterministic checks that confirm the package, configs, legacy command
construction, manifests, and reports are wired correctly before expensive training.

## Create Tiny Datasets

```bash
microi2i create-smoke-data --config configs/smoke/default.yml
```

Outputs:

```text
artifacts/smoke_datasets/
  smoke_dataset_manifest.json
  pix2pix/train/*.png
  pix2pix/val/*.png
  pix2pix/test/*.png
  cyclegan/trainA/*.png
  cyclegan/trainB/*.png
```

The pix2pix images are concatenated `{A,B}` pairs. CycleGAN images are split into separate domains.

## Dry-Run Training

```bash
microi2i train --config configs/train/pix2pix.smoke.yml --dry-run
microi2i train --config configs/train/cyclegan.smoke.yml --dry-run
```

Smoke training configs force:

- CPU execution with `--gpu_ids -1`
- one epoch and no decay epochs
- `max_dataset_size=2`
- image size `32`
- single batch and zero worker subprocesses
- disabled legacy HTML to minimize runtime

## Execute Locally

After verifying dependencies and creating smoke data, remove `--dry-run`:

```bash
microi2i train --config configs/train/pix2pix.smoke.yml
```

The run package includes preflight checks, parsed losses when available, and validation image panels
when legacy training produces sample images.
