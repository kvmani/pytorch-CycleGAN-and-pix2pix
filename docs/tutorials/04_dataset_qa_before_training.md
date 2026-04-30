# Dataset QA Before Training

Dataset QA should run before expensive training jobs.
It provides a machine-readable report and a human-readable contact sheet so obvious data problems are found early.

## Command

```bash
microi2i data-qa --config configs/dataset_qa.default.yml \
  --set source_roots='["D:/microscopy/source"]' \
  --set dataset_id=my_microscopy_dataset \
  --set metadata.modality=SEM \
  --set metadata.material="zirconium alloy"
```

## Outputs

```text
artifacts/dataset_qa/my_microscopy_dataset/
  dataset_qa_report.json
  dataset_qa_report.html
  contact_sheet.jpg
```

## What To Look For

- Failed status means at least one error must be fixed before training.
- Duplicate warnings may be acceptable for controlled repeats, but should be explained.
- Shape mismatch warnings mean resizing/cropping policy should be explicit.
- Leakage-group warnings mean train/validation/test splits may overestimate quality.
