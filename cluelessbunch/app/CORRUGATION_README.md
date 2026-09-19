# Rail Corrugation model

## Active model

RailGuard classifies each one-second, 129-column recording as `Normal`, `Side I`, or
`Side II`. The app and packaged rail predictions use a **75% full-sensor / 25%
sensor-view-augmented linear SVM ensemble**, not the earlier single SVM.

The active artifact is `artifacts/corrugation/model.joblib`, with its settings and promotion
record in `artifacts/corrugation/metadata.json`. The submission snapshot is under
`cluelessbunch/Optional_Items/Rail Corrugation/model/`; its `ensemble_submission/` subfolder
preserves the original pre-promotion artifact and metadata.

## Performance and selection

Macro-F1 gives Normal, Side I and Side II equal importance, despite their unequal frequency.

Local mean outer-fold macro-F1 was **0.7543** for the ensemble versus **0.7554** for the
macro-F1-first single-SVM control: essentially a tie, not a convincing improvement.

Training and hyperparameter tuning used labelled training data only. The final choice to
promote the ensemble used submission feedback. Repeatedly inspected validation folds provide
development evidence, not an independent final evaluation.
Side I remains the weakest class: both approaches missed all three Side I files in the
15+ m/s speed holdout. These results are not a maintenance-safety guarantee.

## How it works

1. Validate the expected 10,000 samples and 129 columns, identifying sensor positions and sides
   from headers. Tachometer readings provide speed diagnostics.
2. Summarise vibration and shock readings on each side. The selected model uses **time-domain
   features**, including signal strength, peaks and variation. The extractor also supports
   spectral and wavelength candidates, but those are not the final selected feature family.
3. Represent each recording twice, once from each side's perspective. A shared fault detector
   lets examples from either side contribute to learning the same fault pattern.
4. Fit one class-weighted SVM on complete sensor views. Fit the second on the full view plus
   eight leave-one-car-out views; the nine views share one unit of source-recording weight.
   Validation and inference use the full recording, not augmented copies.
5. Align the second model's score scale using training-only statistics, then combine 75% of
   the first score with 25% of the aligned second score. These are decision scores, not
   calibrated fault probabilities.
6. Apply the selected decision threshold to return Normal or the detected side.

There are 272 training recordings: 234 Normal, 14 Side I and 24 Side II. Byte-identical
recordings form 270 groups. Validation uses five grouped outer folds repeated five times,
with four grouped inner folds. Duplicates and all views of a recording stay together.
The inner search jointly chooses the component specification, blend weight (0, 0.1 or 0.25),
threshold and side bias using macro-F1-first selection.

The saved ensemble uses `feature_family=time`, `C=0.01`, positive-weight multiplier
`0.75`, no raw-speed feature, threshold `0.3216509649018737`, and zero Side I bias.

## Run the saved model

Run all commands below from `cluelessbunch/app`, with the environment installed as described
in its `README.md`. Retraining is not needed for the app.

```powershell
uv run streamlit run app.py
```

Select **Rail Corrugation** and upload CSV recordings or a ZIP containing them. The app presents
a plain-language result and next step, with technical scores under Engineering details.
It offers a downloadable `rail_predictions.csv`.

For batch inference using the active ensemble:

```powershell
uv run python scripts/predict_corrugation.py `
  --input "../../NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Rail_Corrugation/Test" `
  --output reports/corrugation/predictions_preview.csv `
  --diagnostics reports/corrugation/test_input_diagnostics.csv
```

Replace the input path if your data is elsewhere. The original datasets are excluded from Git.
This example writes a preview, leaving the packaged submission unchanged. Prediction CSVs contain
exactly `file_id,prediction`; diagnostics stay separate. The team's submission is
`cluelessbunch/predictions.zip`, with `rail_predictions.csv` directly at its ZIP root.

## Reproduce the ensemble experiment

These commands require the original labelled training data. Training scripts auto-detect
`NebulaX-Hackathon-ProblemStatement/PS3` at the app or repository root, or use the absolute
directory supplied in `RAILGUARD_PARTICIPANT_ROOT`. Explicit prediction `--input` paths must
still point to the actual input location.

```powershell
uv run python scripts/test_corrugation_ensemble.py
```

This runs the matched nested-validation comparison and speed-holdout checks, writing to
`reports/corrugation_ensemble/`. It does not replace the app artifact or read competition
Test files. The blend weight is reselected within each fold; the reported validation score
evaluates that selection procedure rather than fixing 25% in every fold.

For a separate refit and prediction export:

```powershell
uv run python scripts/predict_corrugation_ensemble.py
```

This requires the local `../Optional_Items/predictions/rail_predictions.csv` as the previous
output to preserve, plus the training and Test datasets. On a fresh clone, the previous CSV can
be extracted from `../predictions.zip`. Archive or rename an existing
`../Optional_Items/predictions/rail_predictions_ensemble.csv` first: the script refuses to
overwrite it. It reselects the ensemble on training data, fits it, then uses Test files for
inference only. It saves the new CSV and a timestamped artifact, metadata and comparison under
`reports/corrugation_ensemble/submissions/`. It does not promote the model or submit online.
Historical paths in archived metadata record the original run location, before repo reorganisation.

## Baseline training is not ensemble training

**Do not run `train_corrugation.py` with its default output directory to reproduce the active
ensemble:** it trains a single SVM and would overwrite the active model. Use isolated outputs:

```powershell
uv run python scripts/train_corrugation.py `
  --artifact-dir reports/corrugation_baseline/artifacts `
  --report-dir reports/corrugation_baseline
uv run python scripts/validate_corrugation.py `
  --model reports/corrugation_baseline/artifacts/model.joblib `
  --report-dir reports/corrugation_baseline
```

`validate_corrugation.py` evaluates a single-SVM specification, not the ensemble blend.
Use `test_corrugation_ensemble.py` for ensemble development validation.

## Historical experiments

These are earlier development comparisons, not descriptions of the active artifact.
All numbers below are local mean outer-fold macro-F1.

| Experiment | Score | Matched control | Script |
|---|---:|---:|---|
| Original single SVM | 0.7419 | — | `train_corrugation.py` |
| Distance-domain features | 0.7375 | 0.7419 | `train_corrugation_v2.py` |
| Standalone sensor-view augmentation | 0.7399 | 0.7419 | `train_corrugation_augmented.py` |
| Macro-F1-first selection | 0.7554 | 0.7419 | `test_corrugation_objective.py` |
| Broader threshold search | 0.7381 | 0.7554 | `test_corrugation_thresholds.py` |
| Extra source-class weighting | 0.7508 | 0.7554 | `test_corrugation_weights.py` |
| Car-aware features | 0.7180 | 0.7554 | `test_corrugation_cars.py` |
| Complementary ensemble | 0.7543 | 0.7554 | `test_corrugation_ensemble.py` |

The ensemble initially showed no convincing local gain. It was subsequently promoted following
the team's reported submission result. Previous active files were backed up under the local,
Git-ignored `reports/corrugation_ensemble/promotions/` directory.

## Tests

```powershell
uv run pytest tests/corrugation
```

Tests that use supplied training data require the excluded datasets; see the app README for
the self-contained test command. Only load trusted Joblib artifacts.
