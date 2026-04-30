# Train And Infer With CycleGAN

## Prepare Dataset

```bash
microi2i prepare-dataset --config configs/dataset_prepare/unaligned_microscopy.default.yml \
  --set source_roots='["D:/microscopy/domain_a","D:/microscopy/domain_b"]' \
  --set output_dataset_dir=D:/microscopy/prepared_cyclegan
```

## Train

```bash
microi2i train --config configs/train/cyclegan.default.yml \
  --set training.legacy_args='["--dataroot","D:/microscopy/prepared_cyclegan","--name","microscopy_cyclegan","--model","cycle_gan","--dataset_mode","unaligned","--gpu_ids","0"]'
```

## Infer

```bash
microi2i infer --config configs/inference/folder.default.yml \
  --set inference.legacy_args='["--dataroot","D:/microscopy/prepared_cyclegan/testA","--name","microscopy_cyclegan","--model","test","--no_dropout","--gpu_ids","-1"]'
```

Use evaluation only when same-named target images exist.
