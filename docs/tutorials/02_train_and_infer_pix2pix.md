# Train And Infer With pix2pix

## Train

```bash
microi2i train --config configs/train/pix2pix.default.yml \
  --set training.legacy_args='["--dataroot","D:/microscopy/prepared_pix2pix","--name","microscopy_pix2pix","--model","pix2pix","--dataset_mode","aligned","--direction","BtoA","--gpu_ids","0"]'
```

Use `--dry-run` first to verify the resolved command and manifests without launching training.

## Infer

```bash
microi2i infer --config configs/inference/folder.default.yml \
  --set inference.legacy_args='["--dataroot","D:/microscopy/infer_inputs","--name","microscopy_pix2pix","--model","test","--no_dropout","--gpu_ids","-1"]'
```

The inference run writes a normalized `report.json` and copies available prediction images into the run package when `inference.expected_output_dir` points to a completed legacy output folder.
