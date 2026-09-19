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

## Experimental v2

The repository also contains a train-only experiment that reconstructs cumulative wheel travel
from tachometer transitions and extracts compact distance-domain wavelength features:

```powershell
uv run python scripts/train_corrugation_v2.py
uv run python scripts/validate_corrugation_v2.py
```

Its outputs are isolated under `artifacts/corrugation_v2`, `cache/corrugation_v2`, and
`reports/corrugation_v2`. The experiment did **not** pass the promotion gates: its mean fold
macro-F1 was 0.7375 versus 0.7419 for v1, and the paired file bootstrap favoured v2 in only
36.35% of resamples. The deployed app therefore continues to use
`artifacts/corrugation/model.joblib`. This rejected experiment is retained for reproducibility
and as a basis for future work if more labelled fault recordings become available.

## Experimental sensor-view augmentation

The repository also contains a fold-safe augmentation experiment. It retains the full recording
and creates eight leave-one-car-out sensor views for training, with the nine correlated views
sharing one unit of sample weight:

```powershell
uv run python scripts/train_corrugation_augmented.py
```

Outputs are isolated under `artifacts/corrugation_aug`, `cache/corrugation_aug`, and
`reports/corrugation_aug`. Validation files are never augmented and every view remains grouped
with its source recording. This experiment was also rejected: mean outer-fold macro-F1 was
0.7399 versus 0.7419 for v1. Side I F1 improved from 0.4645 to 0.4971, but Normal and Side II
weakened, the bootstrap median delta was negative, and high-speed Side I recall remained zero.
The active application model therefore remains unchanged.

## Macro-F1 selection experiment

Run the controlled comparison of recall-constrained and macro-F1-first threshold selection:

```powershell
python scripts/test_corrugation_objective.py
```

Use the project's Python environment. Outputs go to `reports/corrugation_objective/`.
Both policies share identical inner fitted models, candidate grids, and grouped outer folds.
Only recall-gate precedence changes. The baseline reproduced all 1,360 original repeated
validation predictions exactly. Mean fold macro-F1 improved from 0.7419 to 0.7554; pooled
macro-F1 improved from 0.7504 to 0.7673. Side I precision increased from 0.4235 to 0.5000,
with recall unchanged at 0.5143. This is a promising development result, but the +0.0135
mean-fold gain is below the predeclared +0.02 replacement target. The experiment does not
save or deploy a replacement model, read competition test data, or change submission files.

The subsequent broader decision search can be reproduced with:

```powershell
python scripts/test_corrugation_thresholds.py
```

It compares both preceding policies with a symmetric Side I bias grid and all distinct inner
score cutoffs. Results are saved under `reports/corrugation_thresholds/`. The extension was
rejected: mean fold macro-F1 fell to 0.7381 (smaller macro-F1-first search: 0.7554). Side II F1
fell from 0.8246 to 0.7764. Both control policies reproduced their prior predictions exactly.

## Source-class weight experiment and speed checks

```powershell
python scripts/test_corrugation_weights.py
```

This compares Side I source factors 1.0, 1.4 and 1.7, with all choices made inside grouped
inner validation. Reports are stored under `reports/corrugation_weights/`. Mean fold macro-F1
was 0.7508 versus 0.7554 for the unweighted macro-F1-first control, so additional weighting
was rejected. The factor 1.0 was selected in 21/25 outer folds.

The script also reselects every model setting within each speed-holdout training partition.
All policies missed the three Side I files in the 15+ m/s holdout. This stricter diagnostic
differs from older fixed-specification holdouts; the deployed model and submissions are unchanged.

## Car-aware feature experiment

```powershell
python scripts/test_corrugation_cars.py
```

This adds 18 features describing left-right differences within cars and their consistency
across cars, using the same SVM and macro-F1-first selection grid. Reports and feature caches
are isolated under `reports/corrugation_cars/` and `cache/corrugation_cars/`.

The experiment was rejected: mean fold macro-F1 fell from 0.7554 to 0.7180, and pooled
Side I F1 fell from 0.5070 to 0.4503. All five complete-repeat scores declined. Both policies
still missed all three Side I files in the 15+ m/s holdout. The control reproduced all 1,360
previous repeated predictions exactly. All 84 repository tests passed. No competition test
data, deployed artifact or submission was changed. These reused validation folds provide
development evidence, not an independent final-test performance estimate.

## Complementary sensor-view ensemble experiment

```powershell
python scripts/test_corrugation_ensemble.py
```

Blends the full-sensor SVM with a sensor-view-augmented SVM. A shared component specification,
augmented contribution (0, 0.1 or 0.25), threshold and side bias are selected jointly inside
grouped inner validation. Auxiliary margins are aligned using training-only score statistics;
they are not calibrated probabilities. Reports go to `reports/corrugation_ensemble/`.

Mean fold macro-F1 was 0.7543 versus the macro-first control's 0.7554. Pooled Side I recall
rose from 0.5143 to 0.5429, but precision fell from 0.5000 to 0.4810 and Side II F1 declined.
Bootstrap uncertainty spans zero; no convincing gain was measured. Zero auxiliary weight won
16/25 folds, and the three high-speed Side I files were still missed. All 1,360 control
predictions reproduced the prior experiment exactly. All 86 tests passed. The experiment
does not save a replacement artifact, access competition Test data or change submissions.

To explicitly generate a separate submission from the train-selected ensemble:

```powershell
python scripts/predict_corrugation_ensemble.py
```

This separate export fits on all labelled training files, then reads Test files for inference
only. It preserves `rail_predictions.csv` and creates `rail_predictions_ensemble.csv`, refusing
to overwrite an existing ensemble output. A timestamped folder under
`reports/corrugation_ensemble/submissions/` stores the previous CSV, new CSV (with the standard
`rail_predictions.csv` filename), fitted experimental artifact, diagnostics, changed labels,
and metadata/checksums. It does not switch the app's active model or submit files online.
## Current active rail model

The 75% full-sensor / 25% sensor-view SVM ensemble is now active, following user approval
based on reported submission macro-F1 of 68% versus 60% previously. These are user-reported
submission scores, not local validation scores. Local mean fold macro-F1 was 0.7543 versus
0.7554 for the macro-first control. Hyperparameters were selected from training data only;
the final promotion decision uses submission feedback. Historical experiments below retain
their original decisions. Previous model and prediction files are preserved under
`reports/corrugation_ensemble/promotions/`.
