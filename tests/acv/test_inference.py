from pathlib import Path

import joblib

from railguard.acv.config import AcvConfig
from railguard.acv.inference import predict_acv_files, write_predictions_atomic
from railguard.acv.models import FixedPhysicsRanker


def test_inference_and_atomic_output(acv_workbook, tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": FixedPhysicsRanker(),
            "metadata": {"subsystem": "ACV", "config": AcvConfig().__dict__},
        },
        artifact,
    )
    predictions, diagnostics = predict_acv_files([acv_workbook], artifact)
    assert predictions.loc[0, "ranked_cars"].split("|")[0] == "03"
    assert len(diagnostics) == 4
    output = tmp_path / "predictions.csv"
    write_predictions_atomic(predictions, output)
    assert output.is_file()
