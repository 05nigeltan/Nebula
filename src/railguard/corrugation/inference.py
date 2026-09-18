"""Saved-artifact batch inference and atomic submission writing."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.features import extract_features
from railguard.corrugation.metric import validate_prediction_frame
from railguard.corrugation.parsing import CorrugationDataError, load_corrugation_file


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def load_corrugation_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CorrugationDataError(f"Corrugation model artifact does not exist: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"model", "metadata"}.issubset(artifact):
        raise CorrugationDataError("Corrugation artifact has an invalid structure")
    metadata = artifact["metadata"]
    if metadata.get("subsystem") != "Rail Corrugation" or "config" not in metadata:
        raise CorrugationDataError("Artifact is not a compatible Corrugation model")
    return artifact


def predict_corrugation_files(
    input_files: Iterable[str | Path],
    artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted((Path(path) for path in input_files), key=_natural_key)
    if not paths:
        raise CorrugationDataError("No Corrugation CSV files were supplied")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise CorrugationDataError("Corrugation input filenames must be unique")
    if any(path.suffix.lower() != ".csv" for path in paths):
        raise CorrugationDataError("Every Corrugation input must be a CSV file")

    artifact = load_corrugation_artifact(artifact_path)
    config = CorrugationConfig(**artifact["metadata"]["config"])
    rows = []
    for position, path in enumerate(paths, start=1):
        signal = load_corrugation_file(path, config.expected_samples)
        rows.append(extract_features(signal, config))
        print(f"Extracted inference features {position:03d}/{len(paths)}: {path.name}", flush=True)
    features = pd.DataFrame(rows)
    model = artifact["model"]
    side_i_score, side_ii_score = model.decision_scores(features)
    predicted = model.predict(features)
    output = pd.DataFrame({"file_id": names, "prediction": predicted})
    validate_prediction_frame(output)
    diagnostics = pd.DataFrame(
        {
            "file_id": names,
            "speed_mps": features["speed_mps"],
            "tach_transitions": features["tach_transitions"].astype(int),
            "wavelength_valid": features["wavelength_valid"].astype(bool),
            "side_i_score": side_i_score,
            "side_i_adjusted_score": side_i_score + model.spec.side_i_bias,
            "side_ii_score": side_ii_score,
            "decision_threshold": model.spec.threshold,
            "side_i_bias": model.spec.side_i_bias,
        }
    )
    # np.maximum is expressed explicitly to avoid pandas index alignment surprises.
    diagnostics["margin_above_threshold"] = (
        pd.DataFrame({"i": side_i_score + model.spec.side_i_bias, "ii": side_ii_score}).max(axis=1)
        - model.spec.threshold
    )
    return output, diagnostics


def predict_corrugation_directory(
    input_dir: str | Path,
    artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(input_dir)
    if not directory.is_dir():
        raise CorrugationDataError(f"Corrugation input directory does not exist: {directory}")
    return predict_corrugation_files(directory.glob("*.csv"), artifact_path)


def write_predictions_atomic(frame: pd.DataFrame, output_path: str | Path) -> None:
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
