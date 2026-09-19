"""Refresh optional snapshots and predictions from the canonical team/app tree."""

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def main():
    app = Path(__file__).resolve().parents[1]
    team = app.parent
    optional = team / "Optional_Items"
    prediction_dir = optional / "predictions"
    names = [f"{s}_predictions.csv" for s in ("door", "shm", "rail", "acv")]
    for name in names:
        if not (prediction_dir / name).is_file():
            raise FileNotFoundError(prediction_dir / name)
    destination = team / "predictions.zip"
    same = False
    if destination.exists():
        with zipfile.ZipFile(destination) as archive:
            same = set(archive.namelist()) == set(names) and all(
                archive.read(name) == (prediction_dir / name).read_bytes() for name in names
            )
        if not same:
            backup = (
                app / "reports/submission_backups" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            )
            backup.mkdir(parents=True)
            shutil.copy2(destination, backup / "predictions.zip")
    if not same:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(prediction_dir / name, arcname=name)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    for display, module in {
        "Door": "door",
        "SHM": "shm",
        "ACV": "acv",
        "Rail Corrugation": "corrugation",
    }.items():
        for path in (app / "src/railguard" / module).glob("*.py"):
            target = optional / display / "code" / module / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name in ("model.joblib", "metadata.json"):
            target = optional / display / "model" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(app / "artifacts" / module / name, target)
    shutil.copy2(app / "RAILGUARD_TECHNICAL_WRITEUP.md", optional / "write_up.md")
    for name in (
        "DOOR_README.md",
        "SHM_README.md",
        "ACV_README.md",
        "CORRUGATION_README.md",
        "OPERATOR_UI_README.md",
    ):
        shutil.copy2(app / name, optional / "documentation" / name)
    manifest_path = optional / "packaging_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    video = team / "demo_video.mov"
    manifest.update(
        {
            "team": team.name,
            "demo_video": video.name if video.is_file() else "MISSING",
            "demo_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()
            if video.is_file()
            else None,
            "canonical_app_source": "app/",
            "google_cloud_url": "https://railguard-rilk77egmq-uc.a.run.app/",
            "prediction_sources": {name: f"Optional_Items/predictions/{name}" for name in names},
            "prediction_sha256": {
                name: hashlib.sha256((prediction_dir / name).read_bytes()).hexdigest()
                for name in names
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Submission refreshed: {team}; video present: {video.is_file()}")


if __name__ == "__main__":
    main()
