from pathlib import Path

import pandas as pd

from railguard.shm.config import ShmConfig
from railguard.shm.metric import OUTPUT_COLUMNS, validate_prediction_frame
from railguard.shm.parsing import load_training_manifest
from railguard.shm.training import feature_cache_signature


def test_supplied_training_manifest_has_64_independent_files() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM"
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    assert len(manifest) == 64
    assert manifest["filename"].nunique() == 64


def test_example_submission_has_required_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    example = pd.read_csv(
        root
        / "NebulaX-Hackathon-ProblemStatement"
        / "PS3"
        / "04_Example_Submission"
        / "shm_predictions.csv"
    )
    assert tuple(example.columns) == OUTPUT_COLUMNS
    validate_prediction_frame(example)


def test_feature_cache_signature_changes_with_feature_configuration() -> None:
    original = feature_cache_signature(ShmConfig())
    changed = feature_cache_signature(ShmConfig(exponents=(4.0, 5.0)))
    assert original != changed
