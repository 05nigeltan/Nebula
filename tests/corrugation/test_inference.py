from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.features import extract_features
from railguard.corrugation.inference import predict_corrugation_files, write_predictions_atomic
from railguard.corrugation.models import FittedCorrugationModel, ModelSpec
from railguard.corrugation.parsing import load_corrugation_file


def test_saved_artifact_prediction_round_trip(corrugation_csv: Path, tmp_path: Path) -> None:
    config = CorrugationConfig(expected_samples=256)
    base = extract_features(load_corrugation_file(corrugation_csv, 256), config)
    training = pd.DataFrame([base] * 8)

    side_i_column = next(column for column in training if column.startswith("side_i_"))
    side_ii_column = side_i_column.replace("side_i_", "side_ii_", 1)
    training.loc[[4, 6], side_i_column] += np.asarray([3.0, 4.0])
    training.loc[[5, 7], side_ii_column] += np.asarray([3.0, 4.0])
    labels = np.asarray(
        ["Normal", "Normal", "Normal", "Normal", "Side I", "Side II", "Side I", "Side II"]
    )
    model = FittedCorrugationModel(ModelSpec("all", 0.1, 1.0, False)).fit(training, labels)

    artifact_path = tmp_path / "corrugation.joblib"
    joblib.dump(
        {
            "model": model,
            "metadata": {"subsystem": "Rail Corrugation", "config": asdict(config)},
        },
        artifact_path,
    )

    predictions, diagnostics = predict_corrugation_files([corrugation_csv], artifact_path)
    assert predictions["file_id"].tolist() == [corrugation_csv.name]
    assert predictions.loc[0, "prediction"] in {"Normal", "Side I", "Side II"}
    assert diagnostics.loc[0, "file_id"] == corrugation_csv.name
    assert np.isfinite(diagnostics.loc[0, "margin_above_threshold"])

    output_path = tmp_path / "predictions.csv"
    write_predictions_atomic(predictions, output_path)
    pd.testing.assert_frame_equal(pd.read_csv(output_path), predictions)
