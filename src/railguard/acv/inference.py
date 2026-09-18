"""ACV artifact loading, workbook inference, and atomic submission writing."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from railguard.acv.config import AcvConfig
from railguard.acv.features import extract_case_features
from railguard.acv.metric import validate_prediction_frame
from railguard.acv.models import ranking_from_scores
from railguard.acv.parsing import AcvDataError, load_acv_case


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def load_acv_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise AcvDataError(f"ACV model artifact does not exist: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"model", "metadata"}.issubset(artifact):
        raise AcvDataError("ACV artifact has an invalid structure")
    metadata = artifact["metadata"]
    if metadata.get("subsystem") != "ACV" or "config" not in metadata:
        raise AcvDataError("Artifact is not a compatible ACV model")
    return artifact


def predict_acv_files(
    input_files: Iterable[str | Path],
    artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted((Path(path) for path in input_files), key=_natural_key)
    if not paths:
        raise AcvDataError("No ACV Excel workbooks were supplied")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise AcvDataError("ACV input filenames must be unique")
    if any(path.suffix.lower() != ".xlsx" for path in paths):
        raise AcvDataError("Every ACV input must be an .xlsx workbook")

    artifact = load_acv_artifact(artifact_path)
    config = AcvConfig(**artifact["metadata"]["config"])
    model = artifact["model"]
    output_rows = []
    diagnostic_frames = []
    for path in paths:
        case = load_acv_case(path)
        features = extract_case_features(case, config)
        scores = np.asarray(model.score(features), dtype=float)
        ranking = ranking_from_scores(features, scores)
        output_rows.append({"file_id": case.file_id, "ranked_cars": "|".join(ranking)})
        local = features.copy()
        local["fault_score"] = scores
        local["predicted_rank"] = local["car"].map(
            {car: index for index, car in enumerate(ranking, start=1)}
        )
        ordered_scores = np.sort(scores)[::-1]
        margin = float(ordered_scores[0] - ordered_scores[1]) if len(scores) > 1 else np.nan
        local["top_score_margin"] = margin
        local["evidence_separation"] = np.select(
            [
                ~local["supported"],
                local["supported"] & (margin >= config.confidence_margin),
            ],
            ["Insufficient data", "Clear lead"],
            default="Close call",
        )
        # Retained for compatibility with earlier diagnostics exports.
        local["confidence"] = np.where(
            local["evidence_separation"].eq("Clear lead"), "higher", "low"
        )
        diagnostic_frames.append(local.sort_values("predicted_rank"))
    output = pd.DataFrame(output_rows)
    validate_prediction_frame(output)
    return output, pd.concat(diagnostic_frames, ignore_index=True)


def predict_acv_directory(
    input_dir: str | Path,
    artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(input_dir)
    if not directory.is_dir():
        raise AcvDataError(f"ACV input directory does not exist: {directory}")
    return predict_acv_files(directory.glob("*.xlsx"), artifact_path)


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
        written = pd.read_csv(temporary_name, dtype=str)
        validate_prediction_frame(written)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
