# Inference Batch Review

MicroI2I packages inference outputs into a reviewable scientific run folder.
This is useful when legacy `test.py` or domain scripts generate many images that need inspection.

## Command

```bash
microi2i infer --config configs/inference/folder.default.yml \
  --set inference.expected_output_dir=D:/runs/microscopy_inference/test_latest/images
```

## Output Package

```text
<run_dir>/
  predictions/
  batch_summary.json
  batch_summary.csv
  review.html
  report.json
  report.html
  run_manifest.json
  artifact_manifest.json
```

## Postprocessing

Use postprocessing for standardized review artifacts:

```bash
microi2i infer --config configs/inference/folder.default.yml \
  --set inference.postprocess.grayscale=true \
  --set inference.postprocess.resize=[256,256] \
  --set inference.postprocess.rename_prefix=pred
```

These operations affect the packaged review copy, not the original legacy output folder.
