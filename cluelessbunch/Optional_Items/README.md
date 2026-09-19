# Submission contents

Team: **cluelessbunch**.

The demo video is included as `../demo_video.mov`. Its original QuickTime bytes were preserved;
the extensionless file `demo_vid` was renamed to match its detected container format.

`../predictions.zip` contains exactly these four files directly at the ZIP root:

- `door_predictions.csv`
- `shm_predictions.csv`
- `rail_predictions.csv` — generated with the active 75% full-sensor / 25% sensor-view SVM ensemble
- `acv_predictions.csv`

The ensemble has been promoted to the working project's main rail CSV and active model.
Previous working files were backed up before promotion. The packaging manifest records
the selected prediction sources and their checksums.

`../app/` is the canonical source directory and contains the runnable app, complete Python package, dependency files, configuration,
tests, scripts and four active model artifacts. See its README for setup instructions.

`Door/`, `ACV/`, `Rail Corrugation/` and `SHM/` each contain a `code/` snapshot and a `model/`
folder. The per-subsystem code is provided for inspection; use the complete package under
`../app/src/` for execution and cross-module dependencies. For Rail Corrugation, `model/`
contains the promoted ensemble, while `model/ensemble_submission/` preserves its original
pre-promotion artifact and metadata used for this prediction ZIP.

`write_up.md` describes the promoted ensemble, local validation results and selection limitations.
Further subsystem documentation is in `documentation/`.

Dataset files remain outside the team folder. Virtual environments, caches, reports, historical
archives and individual prediction CSVs are local Git-ignored files; exclude them if manually
archiving the submission. `../predictions.zip` already contains all required scoring CSVs.

Cloud app: https://railguard-rilk77egmq-uc.a.run.app/
