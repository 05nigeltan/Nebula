# RailGuard

Railway condition monitoring by **cluelessbunch** for the NebulaX Hackathon, Problem Statement 3.

RailGuard turns train sensor recordings into clear results and suggested next steps. Upload your
data through one web app to investigate four railway systems:

- **Doors:** detect unusual resistance during opening and closing.
- **Structural Health Monitoring (SHM):** estimate fatigue damage from repeated stress changes.
- **Rail Corrugation:** identify patterns associated with uneven rail surfaces and the affected side.
- **Air Conditioning (ACV):** rank train cars that may have a refrigerant leak.

Results support maintenance review, not a safety clearance or a replacement for engineering judgement.

## Try it

[Open RailGuard on Google Cloud Run](https://railguard-rilk77egmq-uc.a.run.app/)

The hosted demo depends on the hackathon cloud environment remaining active. You can also
[download the demo video](cluelessbunch/demo_video.mov) or run the app locally.

## Submission files

- [Predictions ZIP](cluelessbunch/predictions.zip) — one prediction CSV for each subsystem.
- [App source and setup guide](cluelessbunch/app/README.md) — runnable code and saved models.
- [Technical write-up](cluelessbunch/Optional_Items/write_up.md) — approach, model choices and limitations.
- [Optional materials](cluelessbunch/Optional_Items/README.md) — subsystem code, models and documentation.

## Run locally

With Python 3.12+ and uv installed, run these commands from the repository root:

```sh
cd cluelessbunch/app
uv sync --frozen
uv run streamlit run app.py
```

Saved models are included; no retraining is needed. Upload your own supported recordings.
The original hackathon datasets are excluded from this repository. For setup without uv,
see the [app guide](cluelessbunch/app/README.md). Cloud deployment instructions are in the
[Cloud Run guide](cluelessbunch/app/GOOGLE_CLOUD_DEPLOYMENT.md).

Built with Python, Streamlit, scikit-learn, Pandas, NumPy, SciPy and Plotly; packaged with
Docker for Google Cloud Run.
