from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from railguard.shm.config import ShmConfig
from railguard.shm.inference import predict_shm_directory, write_predictions_atomic
from railguard.shm.models import FittedDamageModel, PhysicsSpec, ResidualSpec
from railguard.shm.parsing import load_shm_file
from railguard.shm.rainflow_features import extract_features


def _small_artifact(tmp_path: Path, input_dir: Path) -> Path:
    config = ShmConfig(expected_samples=5)
    rows = []
    targets = []
    for index, amplitude in enumerate((2.0, 4.0), start=1):
        path = input_dir / f"fit{index}.csv"
        np.savetxt(path, [0.0, amplitude, 0.0, amplitude, 0.0])
        rows.append(extract_features(load_shm_file(path, 5), config))
        targets.append(float(amplitude**5))
    frame = pd.DataFrame(rows)
    model = FittedDamageModel(PhysicsSpec(5.0, "fixed"), ResidualSpec("none"), config).fit(
        frame, np.asarray(targets)
    )
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {"model": model, "metadata": {"subsystem": "SHM", "config": config.__dict__}},
        artifact,
    )
    return artifact


def test_directory_inference_is_naturally_sorted_and_positive(tmp_path: Path) -> None:
    input_dir = tmp_path / "signals"
    input_dir.mkdir()
    artifact = _small_artifact(tmp_path, input_dir)
    (input_dir / "fit1.csv").rename(input_dir / "test10.csv")
    (input_dir / "fit2.csv").rename(input_dir / "test2.csv")
    predictions, diagnostics = predict_shm_directory(input_dir, artifact)
    assert predictions["file_id"].tolist() == ["test2.csv", "test10.csv"]
    assert np.all(predictions["prediction"] > 0)
    assert np.allclose(diagnostics["correction_factor"], 1.0)


def test_atomic_writer_does_not_replace_destination_on_invalid_input(tmp_path: Path) -> None:
    destination = tmp_path / "predictions.csv"
    destination.write_text("sentinel", encoding="utf-8")
    invalid = pd.DataFrame({"file_id": ["a.csv"], "prediction": [np.nan]})
    with pytest.raises(ValueError):
        write_predictions_atomic(invalid, destination)
    assert destination.read_text(encoding="utf-8") == "sentinel"
