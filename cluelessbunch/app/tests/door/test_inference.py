from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from railguard.door.config import OUTPUT_COLUMNS
from railguard.door.inference import predict_door_detailed
from railguard.door.metric import validate_prediction_frame
from railguard.door.models import OperationThresholdClassifier
from railguard.door.parsing import load_door_csv
from railguard.door.segmentation import segment_stream
from railguard.project_paths import participant_root


def test_example_submission_has_required_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    example = pd.read_csv(participant_root(root) / "04_Example_Submission" / "door_predictions.csv")
    assert tuple(example.columns) == OUTPUT_COLUMNS
    validate_prediction_frame(example)


def test_detailed_inference_exposes_explanation_evidence(two_cycle_csv, tmp_path: Path) -> None:
    segments, _ = segment_stream(load_door_csv(two_cycle_csv))
    model = OperationThresholdClassifier().fit(segments, np.array([0, 1]))
    artifact = tmp_path / "door.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": "threshold_baseline",
            "score_threshold": 0.0,
            "config": {"gap_threshold_ms": 1_000.0, "template_points": 64},
        },
        artifact,
    )
    predictions, segmentation, details = predict_door_detailed(two_cycle_csv, artifact)
    assert len(predictions) == segmentation.cycle_count == len(details) == 2
    assert {"model_score", "score_margin", "current_mean", "operation"}.issubset(details)
