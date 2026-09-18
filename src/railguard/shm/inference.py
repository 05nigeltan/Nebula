"""Saved-artifact SHM batch inference and atomic output writing."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from railguard.shm.config import ShmConfig
from railguard.shm.metric import validate_prediction_frame
from railguard.shm.parsing import ShmDataError, load_shm_file
from railguard.shm.rainflow_features import extract_features


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def load_shm_artifact(artifact_path: str | Path) -> dict[str, Any]:
    path = Path(artifact_path)
    if not path.is_file():
        raise ShmDataError(f"SHM model artifact does not exist: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"model", "metadata"}.issubset(artifact):
        raise ShmDataError("SHM model artifact has an invalid structure")
    metadata = artifact["metadata"]
    if metadata.get("subsystem") != "SHM" or "config" not in metadata:
        raise ShmDataError("Artifact is not a compatible SHM model")
    return artifact


def predict_shm_files(
    input_files: Iterable[str | Path],
    artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict a validated collection of CSV files in deterministic order."""

    paths = sorted((Path(path) for path in input_files), key=_natural_key)
    if not paths:
        raise ShmDataError("No SHM CSV files were supplied")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ShmDataError("SHM input filenames must be unique")
    if any(path.suffix.lower() != ".csv" for path in paths):
        raise ShmDataError("Every SHM input must be a CSV file")

    artifact = load_shm_artifact(artifact_path)
    config_values = dict(artifact["metadata"]["config"])
    config = ShmConfig(**config_values)
    rows = []
    for path in paths:
        signal = load_shm_file(path, config.expected_samples)
        rows.append(extract_features(signal, config))
    features = pd.DataFrame(rows)
    model = artifact["model"]
    predictions = model.predict(features)
    physics, correction = model.predict_components(features)
    output = pd.DataFrame({"file_id": names, "prediction": predictions})
    validate_prediction_frame(output)
    diagnostics = pd.DataFrame(
        {
            "file_id": names,
            "sample_count": features["sample_count"].astype(int),
            "physics_prediction": physics,
            "correction_factor": correction,
            "signal_rms": features["stat_rms"],
            "maximum_cycle_range": features["rf_max_range"],
            "top_1pct_damage_share": features["rf_top_01_damage_share"],
        }
    )
    return output, diagnostics


def predict_shm_directory(
    input_dir: str | Path,
    artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(input_dir)
    if not directory.is_dir():
        raise ShmDataError(f"SHM input directory does not exist: {directory}")
    return predict_shm_files(directory.glob("*.csv"), artifact_path)


def write_predictions_atomic(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Validate the complete frame before atomically replacing the destination."""

    validate_prediction_frame(frame)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            frame.to_csv(handle, index=False)
        written = pd.read_csv(temporary_name)
        validate_prediction_frame(written)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
