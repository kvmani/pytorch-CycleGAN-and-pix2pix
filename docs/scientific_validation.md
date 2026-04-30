# Scientific Validation

## Validation Philosophy

Image translation outputs must be evaluated as scientific artifacts, not only as visually pleasing images.

## General Image Translation Metrics

- MAE and RMSE for paired reconstruction error.
- PSNR for signal fidelity.
- SSIM for structural similarity.
- Histogram and contrast statistics for intensity distribution changes.
- Edge preservation metrics for boundary-sensitive microscopy tasks.

## pix2pix Validation

Use paired validation/test splits when ground truth exists.
Report aggregate metrics and per-sample outliers.
Keep validation and test splits isolated from training and augmentation leakage.

## CycleGAN Validation

Unpaired translation needs additional caution.
Report:

- Cycle-consistency loss trends.
- Domain-level image statistics before and after translation.
- Expert review panels.
- Downstream task metrics when available.

## EBSD And Kikuchi Validation

Recommended metrics include:

- Pattern sharpness proxies.
- Band contrast proxies.
- Noise reduction without band destruction.
- Downstream indexing or pattern usability metrics when tooling is available.
- Side-by-side review panels for representative materials and acquisition conditions.

## Super-Resolution And Denoising

Recommended metrics include:

- PSNR and SSIM on paired degradation benchmarks.
- Edge preservation.
- Frequency-domain checks where relevant.
- Failure analysis on hallucinated features.

## Required Reporting

Every evaluation run should emit:

- `report.json`
- Optional `report.html`
- Per-sample metric table
- Aggregate metric summary
- Manifest references for dataset, model, and artifacts
