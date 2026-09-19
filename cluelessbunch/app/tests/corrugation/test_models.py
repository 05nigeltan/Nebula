import numpy as np
import pandas as pd

from railguard.corrugation.models import (
    FittedCorrugationModel,
    ModelSpec,
    build_paired_matrix,
    feature_bases,
)
from railguard.corrugation.models_v2 import FittedCorrugationV2, V2ModelSpec


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side_i_vibration_rms_median": [0, 0, 0, 0, 5, 0, 4, 0],
            "side_ii_vibration_rms_median": [0, 0, 0, 0, 0, 5, 0, 4],
            "side_i_shock_peak_q90": [1, 1, 1, 1, 8, 1, 7, 1],
            "side_ii_shock_peak_q90": [1, 1, 1, 1, 1, 8, 1, 7],
            "wavelength_valid": np.ones(8),
            "tach_transitions": np.arange(8),
            "speed_mps": np.arange(8) / 2,
            "tach_duty": np.full(8, 0.5),
        }
    )


def test_paired_matrix_swaps_focal_and_other_side() -> None:
    frame = _feature_frame().iloc[:1]
    bases = feature_bases(frame, "all")
    matrix = build_paired_matrix(frame, bases, include_raw_speed=False)
    assert matrix.shape == (2, len(bases) * 4 + 1)
    np.testing.assert_allclose(matrix[0, : len(bases)], matrix[1, len(bases) : 2 * len(bases)])


def test_side_symmetric_model_produces_valid_labels() -> None:
    frame = _feature_frame()
    labels = np.array(
        ["Normal", "Normal", "Normal", "Normal", "Side I", "Side II", "Side I", "Side II"]
    )
    spec = ModelSpec("all", 0.1, 1.0, False, 0.0, 0.0)
    model = FittedCorrugationModel(spec).fit(frame, labels)
    predicted = model.predict(frame)
    assert set(predicted).issubset({"Normal", "Side I", "Side II"})
    assert len(predicted) == len(frame)


def test_v2_rbf_model_produces_valid_labels() -> None:
    frame = _feature_frame().assign(
        side_i_vibration_spatial_peak_prominence_q90=[0, 0, 0, 0, 4, 0, 3, 0],
        side_ii_vibration_spatial_peak_prominence_q90=[0, 0, 0, 0, 0, 4, 0, 3],
        spatial_valid=np.ones(8),
    )
    labels = np.array(
        ["Normal", "Normal", "Normal", "Normal", "Side I", "Side II", "Side I", "Side II"]
    )
    spec = V2ModelSpec("spatial_time", "rbf", 1.0)
    model = FittedCorrugationV2(spec).fit(frame, labels)
    predicted = model.predict(frame)
    assert set(predicted).issubset({"Normal", "Side I", "Side II"})
    assert len(predicted) == len(frame)
