from pathlib import Path

import pandas as pd

from railguard.corrugation.config import OUTPUT_COLUMNS
from railguard.corrugation.metric import validate_prediction_frame
from railguard.corrugation.parsing import load_training_manifest
from railguard.project_paths import participant_root


def test_supplied_training_manifest_has_expected_classes() -> None:
    root = Path(__file__).resolve().parents[2]
    data = participant_root(root) / "02_Datasets" / "Rail_Corrugation"
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    assert len(manifest) == 272
    assert manifest["label"].value_counts().to_dict() == {
        "Normal": 234,
        "Side II": 24,
        "Side I": 14,
    }


def test_example_submission_has_required_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    example = pd.read_csv(participant_root(root) / "04_Example_Submission" / "rail_predictions.csv")
    assert tuple(example.columns) == OUTPUT_COLUMNS
    validate_prediction_frame(example)
