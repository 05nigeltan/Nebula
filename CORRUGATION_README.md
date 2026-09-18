# Rail Corrugation model

This subsystem classifies each one-second, 129-column axle-box recording as `Normal`, `Side I`,
or `Side II`. It uses a shared side-local linear SVM so that Side I and Side II contribute to the
same fault detector while retaining the required side localization.

## Method

1. Validate the 10,000-row schema and derive every car, position, signal type, and rail side from
   the CSV headers.
2. Convert tachometer transitions to speed using the documented 90-tooth wheel and 0.85 m wheel
   diameter.
3. Extract robust time-domain, Welch spectral, speed-normalized wavelength, short-window
   persistence, and cross-axle consensus features for vibration and shock channels.
4. Represent each source file twice, once with Side I as the focal side and once with Side II.
5. Fit a class-weighted, L2-regularized linear SVM with a three-class decision threshold selected
   inside grouped validation folds.

Byte-identical files are kept in the same fold. Feature family, SVM regularization, class weight,
raw-speed inclusion, and decision threshold are selected inside nested validation.

## Train

```powershell
uv run python scripts/train_corrugation.py
```

Outputs:

- `artifacts/corrugation/model.joblib`
- `artifacts/corrugation/metadata.json`
- `cache/corrugation/train_features.csv`
- `reports/corrugation/training_report.json`
- `reports/corrugation/fold_results.csv`
- `reports/corrugation/out_of_fold_predictions.csv`

Training never accepts or reads a competition test directory.

## Robustness validation

```powershell
uv run python scripts/validate_corrugation.py
```

This evaluates the overlapping-speed subset, speed-bin holdouts, vibration/shock ablations, and
the no-raw-speed model. Results are written to
`reports/corrugation/robustness_report.json`.

## Predict

```powershell
uv run python scripts/predict_corrugation.py `
  --input NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Rail_Corrugation/Test `
  --output rail_predictions.csv `
  --diagnostics reports/corrugation/test_input_diagnostics.csv
```

The submission file contains exactly:

```text
file_id,prediction
Test1.csv,Normal
```

Diagnostics are deliberately kept in a separate file.

## App

```powershell
uv run streamlit run app.py
```

Select **Rail Corrugation**, then upload one or more CSV files or a ZIP containing CSV files. The
app displays the class result, estimated speed, side decision scores, and a downloadable
`rail_predictions.csv`.

## Tests

```powershell
uv run pytest tests/corrugation
```

