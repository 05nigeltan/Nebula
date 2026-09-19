# SHM fatigue-damage model

## Model

The deployed SHM artifact is a calibrated rainflow-Miner model trained on the 64 labelled signal
files. Every complete signal file is one independent example. The selected equation is:

```text
damage = 1.36023185508412e-09 * sum(cycle_count * stress_amplitude ** 5)
```

Nested leave-one-file-out validation produced 2.544% MAPE and an estimated official score of
0.97456. The scale is fitted by the exact weighted-median solution for training-fold MAPE. A
Ridge residual challenger reached lower average MAPE, but failed the consistency,
condition-cluster, and bootstrap gates, so it is not enabled in the saved artifact.

## Commands

Install dependencies and train:

```powershell
uv sync --extra dev --extra app
uv run python scripts/train_shm.py
```

Run train-only robustness checks:

```powershell
uv run python scripts/validate_shm.py
```

Generate a submission file and optional diagnostic report:

```powershell
uv run python scripts/predict_shm.py `
  --input "NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/SHM/Test" `
  --output shm_predictions.csv `
  --diagnostics reports/shm/test_input_diagnostics.csv
```

Run verification and the app:

```powershell
uv run pytest
uv run --extra app streamlit run app.py
```

The app accepts one or more SHM CSV files or a ZIP of CSV files. It displays predictions, a
downsampled signal trace, fifth-power damage contribution by amplitude bin, diagnostics, and a
download button.

## Output contract

`shm_predictions.csv` contains exactly:

```csv
file_id,prediction
```

Inputs must be headerless, finite, single-column numeric CSV files. Prediction order is natural
filename order, and all predictions must be finite and strictly positive. The CLI validates the
complete batch before atomically replacing the requested output file.

## Generated files

```text
artifacts/shm/
|-- model.joblib
`-- metadata.json

reports/shm/
|-- training_report.json
|-- candidate_results.csv
|-- fold_results.csv
|-- out_of_fold_predictions.csv
|-- feature_diagnostics.csv
|-- robustness_report.json
|-- repeated_validation_predictions.csv
|-- condition_cluster_predictions.csv
|-- ablation_results.csv
`-- test_input_diagnostics.csv

cache/shm/
|-- train_features.csv
`-- cache_metadata.json
```

The ignored feature cache is reused only when its extraction signature, ordered file identifiers,
and every source SHA-256 hash match.

## Interpretation limit

This model estimates the supplied rainflow/Miner-style cumulative-damage target. It is not a
remaining-life guarantee and must not be used by itself for a maintenance-safety decision.
