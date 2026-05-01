# Scientific Validation

## Validation Philosophy

Image translation outputs must be evaluated as scientific artifacts, not only as visually pleasing images.

## General Image Translation Metrics

- MAE and RMSE for paired reconstruction error.
- PSNR for signal fidelity.
- SSIM for structural similarity.
- Histogram and contrast statistics for intensity distribution changes.
- Edge preservation metrics for boundary-sensitive microscopy tasks.
- High-frequency energy and Laplacian sharpness proxies for denoising and super-resolution screening.
- Per-sample outlier ranking for expert review.


## Longitudinal Training Validation

```{image} diagrams/training_validation_monitor.svg
:alt: Epoch-level validation monitor workflow
:class: architecture-diagram
```


Training-time validation monitoring uses the same fixed samples across epochs so reviewers can
inspect whether a model is improving, stagnating, or hallucinating features. A small fixed panel is
scientifically useful because visual changes are attributable to the model state rather than sample
selection. When more than five samples are requested, MicroI2I keeps the first five fixed and uses
deterministic epoch-level random sampling for the remaining slots to improve coverage.

For paired pix2pix datasets, monitor reports compute paired metrics for generated and target images.
For unpaired CycleGAN datasets, reports emphasize visual progression unless explicit references are
available; arbitrary unpaired target-domain images are not treated as ground truth.

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
- Best/worst sample review panels when paired predictions and targets are available
- Manifest references for dataset, model, and artifacts

## Metric Families

Evaluation reports now group aggregate metrics into `fidelity`, `structure`, `microscopy`, and `ebsd_kikuchi` families. The grouping is meant for review dashboards and report navigation; it is not an automatic model-ranking rule.

## Current Implemented Metrics

`microi2i evaluate` currently computes the following per same-named prediction/target pair:

- `mae`, `rmse`, `psnr`, and optional `ssim`
- `gradient_correlation`
- `edge_mae`
- `histogram_l1`
- `cnr_proxy_delta`
- `high_frequency_energy_ratio`
- `laplacian_sharpness_ratio`
- `ebsd_band_contrast_delta`
- `ebsd_band_sharpness_ratio`
- `orientation_coherence_delta`

These are screening metrics, not a replacement for microscopy expertise. For EBSD/Kikuchi images,
band preservation and downstream indexing quality remain the stronger validation targets when tooling
and reference data are available.
