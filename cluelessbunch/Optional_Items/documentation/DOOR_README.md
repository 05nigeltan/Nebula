# Door abnormal-resistance pipeline

This project turns the continuous Door sensor stream into exact cycle boundaries and classifies
each cycle as `Normal` or `Abnormal resistance`.

## What is implemented

- Strict validation and native timestamp parsing for all 17 sensor columns.
- Deterministic cycle segmentation at inter-reading gaps greater than 1 second.
- Open/Close inference from controller commands, cross-checked against door position.
- Leakage-safe, operation-specific feature extraction and Normal-current templates.
- Bounded comparison of a current-threshold baseline, two logistic regressions, a linear SVM,
  and shallow gradient boosting.
- Contiguous nested cross-validation, abnormal-recall guardrail, and the exact official
  IoU-weighted F1 metric.
- Saved model artifact, metadata, out-of-fold predictions, required prediction CLI, tests, and a
  small Streamlit demonstration app.

## Run it

Install the locked environment and train:

```powershell
uv sync --extra dev --extra app
uv run python scripts/train_door.py
```

Generate the required submission file from CSV or Excel:

```powershell
uv run python scripts/predict.py `
  --input "NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Door/Test.csv" `
  --output door_predictions.csv
```

For a multi-sheet Excel workbook, select a worksheet in the app or pass its name to the CLI:

```powershell
uv run python scripts/predict.py --input door_readings.xlsx `
  --sheet "Door readings" --output door_predictions.csv
```

Excel input must contain the same 17 Door sensor columns as `Train.csv`. The app rejects files
from other subsystems instead of generating misleading predictions.

Run tests and the optional app:

```powershell
uv run pytest
uv run streamlit run app.py
```

## Outputs

- `artifacts/door/model.joblib`: fitted end-to-end classifier.
- `artifacts/door/metadata.json`: configuration, hashes, model selection, and validation results.
- `reports/door/training_report.json`: machine-readable experiment report.
- `reports/door/out_of_fold_predictions.csv`: auditable validation predictions.
- `door_predictions.csv`: submission-ready predictions for the supplied test stream.

The training labels are used only while fitting each cross-validation fold. Normal templates and
operation-specific scaling are also fitted inside each fold, preventing validation leakage.

## Verified result on the supplied training data

The committed artifact was selected using five contiguous outer folds with nested tuning:

| Candidate | Official IoU-F1 | Accuracy | Abnormal recall |
|---|---:|---:|---:|
| Current threshold | 0.9909 | 0.9909 | 0.9667 |
| Logistic, minimal features | 0.9909 | 0.9909 | 0.9667 |
| Logistic, physics features | 0.9909 | 0.9909 | 0.9667 |
| **Linear SVM, physics features** | **1.0000** | **1.0000** | **1.0000** |
| Shallow gradient boosting | 0.9909 | 0.9909 | 0.9667 |

This is an internal cross-validation estimate, not a claim about the hidden test labels. The
supplied test stream produced 38 segments: 30 predicted Normal and 8 predicted Abnormal
resistance. The output has exactly the three required submission columns.
