# ACV Refrigerant-Leak Localisation

This pipeline ranks every car in an ACV workbook from most to least likely to have a
refrigerant leak. It uses peer-relative cabin-temperature and setpoint deviations during valid
cooling operation. The complete workbook is the statistical unit: timestamp rows are never
randomly divided between training and validation.

## Train and validate

```powershell
uv run python scripts/train_acv.py
uv run python scripts/validate_acv.py
```

The training command writes:

- `artifacts/acv/model.joblib` and `artifacts/acv/metadata.json`
- `reports/acv/training_report.json`
- complete-case out-of-fold rankings, feature ablations, and robustness reports

The five workbooks sharing the test schema drive model selection. The single rich pressure-schema
workbook is reported separately because one case is not enough to validate pressure-specific
learning.

## Predict

```powershell
uv run python scripts/predict_acv.py `
  --input NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/ACV/Test `
  --output acv_predictions.csv `
  --diagnostics reports/acv/test_input_diagnostics.csv
```

The submission file has exactly two columns: `file_id` and `ranked_cars`. Diagnostics include the
per-car fault score, rank, usable-sample count, and a simple score-margin confidence flag. This is
a comparative localisation model, not a calibrated leak probability or a safety-release system.
