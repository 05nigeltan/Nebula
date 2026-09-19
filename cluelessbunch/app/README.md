# RailGuard — cluelessbunch

Streamlit application for Door, SHM, Rail Corrugation and ACV analysis.

## Google Cloud deployment

Live Google Cloud deployment: https://railguard-rilk77egmq-uc.a.run.app/

This directory is now the canonical app source, not a generated copy. From the repository
root, first run `cd cluelessbunch/app`. Cloud Run deployment files are included.
Follow `GOOGLE_CLOUD_DEPLOYMENT.md` and run
`bash scripts/deploy_cloud_run.sh` from this app directory in authenticated Cloud Shell.
The live homepage and health endpoint returned HTTP 200 when checked; that is not an end-to-end
cloud inference test. `python scripts/package_cloud_run.py` creates a separate minimal
`railguard-cloud-run.zip` in this app directory, not the predictions submission ZIP.
Changing the local layout does not redeploy or alter the running cloud service.

Direct uploads now have a 25 MB limit; an optional Cloud Storage demo library supports larger
ACV workbooks. Install the `cloud` extra for this optional feature when running outside Docker.

## Run locally

Use Python 3.12 or newer. From this `app` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, with uv installed: `uv sync --frozen`, then `uv run streamlit run app.py`.
Upload your own supported recordings through the UI. The original hackathon datasets are
not included. The source package and four active model artifacts are included, so retraining
is not required to run the app. Training scripts require the original labelled datasets.

## Active rail model

The team's `predictions.zip` and active app both use the 75% full-sensor /
25% sensor-view ensemble. Hyperparameters were selected using training-only validation;
the team approved promotion using submission feedback.
The active ensemble artifact and metadata are under `artifacts/corrugation/`, mirrored in
`../Optional_Items/Rail Corrugation/model/`. The latter's `ensemble_submission/` subfolder
preserves the original pre-promotion artifact and metadata, not the current promotion record.
See `CORRUGATION_README.md` for the method, results and safe reproduction commands.

The ensemble did not demonstrate a validation improvement (mean fold macro-F1 0.7543 versus
0.7554 for the macro-F1-first control). Submission feedback informed the final model choice
and is not an independent final evaluation. All outputs support review rather than replace approved
engineering or maintenance procedures. Only load trusted Joblib artifacts.

## Tests

Install development tools with `python -m pip install -e ".[dev]"`. Seven tests require
the excluded hackathon training datasets or example submission files; a plain full test run
will fail those seven until the original
`NebulaX-Hackathon-ProblemStatement` directory is supplied at this app root or the repository
root. Alternatively set `RAILGUARD_PARTICIPANT_ROOT` to the absolute path of its `PS3` directory.
In this working tree the original datasets remain at repository root and are auto-detected.

Canonical individual prediction CSVs are kept locally under `../Optional_Items/predictions/`.
Run `python scripts/package_submission.py` to refresh optional code/model snapshots and the
team-level prediction ZIP. Existing identical ZIPs are preserved; changed ZIPs are backed up.
Caches, reports, raw datasets and virtual environments are Git-ignored development files,
not part of the repository submission. Do not manually upload those local folders.

Run only the self-contained tests with:

```powershell
python -m pytest -q -k "not supplied_training and not example_submission"
```
