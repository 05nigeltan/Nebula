"""Saved-artifact inference for Door streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from railguard.door.config import ABNORMAL_LABEL, NORMAL_LABEL
from railguard.door.features import extract_cycle_features
from railguard.door.metric import validate_prediction_frame
from railguard.door.models import model_scores
from railguard.door.parsing import load_door_data
from railguard.door.segmentation import SegmentationDiagnostics, segment_stream


def predict_door(
    input_data: str | Path | Any,
    artifact_path: str | Path,
    *,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, SegmentationDiagnostics]:
    """Run the saved Door pipeline and return submission-ready predictions."""

    output, diagnostics, _ = predict_door_detailed(
        input_data,
        artifact_path,
        sheet_name=sheet_name,
    )
    return output, diagnostics


def predict_door_detailed(
    input_data: str | Path | Any,
    artifact_path: str | Path,
    *,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, SegmentationDiagnostics, pd.DataFrame]:
    """Return submission output plus auditable per-cycle explanation diagnostics."""

    artifact = joblib.load(artifact_path)
    config = artifact["config"]
    frame = load_door_data(input_data, sheet_name=sheet_name)
    segments, diagnostics = segment_stream(
        frame, gap_threshold_ms=float(config["gap_threshold_ms"])
    )
    model = artifact["model"]
    if artifact["model_name"] == "threshold_baseline":
        scores = np.asarray(model.decision_function(segments), dtype=float)
        score_threshold = 0.0
        predicted = model.predict(segments)
    else:
        scores = model_scores(model, segments)
        score_threshold = float(artifact["score_threshold"])
        predicted = (scores >= score_threshold).astype(int)
    output = pd.DataFrame(
        {
            "start_time": [segment.start_time for segment in segments],
            "end_time": [segment.end_time for segment in segments],
            "prediction": np.where(predicted == 1, ABNORMAL_LABEL, NORMAL_LABEL),
        }
    )
    validate_prediction_frame(output)

    transformer = getattr(model, "named_steps", {}).get("features")
    template_points = int(config["template_points"])
    detail_rows = []
    for segment, score in zip(segments, scores, strict=True):
        template = (
            getattr(transformer, "templates_", {}).get(segment.operation)
            if transformer is not None
            else None
        )
        features = extract_cycle_features(
            segment,
            normal_template=template,
            template_points=template_points,
        )
        detail_rows.append(
            {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "operation": segment.operation,
                "model_score": float(score),
                "detection_threshold": score_threshold,
                "score_margin": float(score - score_threshold),
                "duration_s": features["duration_s"],
                "current_mean": features["current_mean"],
                "current_p95": features["current_p95"],
                "mid_current_mean": features["mid_current_mean"],
                "template_residual_mean": features["template_residual_mean"],
                "operation_warning": segment.operation_warning,
            }
        )
    return output, diagnostics, pd.DataFrame(detail_rows)
