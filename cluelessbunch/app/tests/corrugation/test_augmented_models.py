from __future__ import annotations

import numpy as np
import pandas as pd

from railguard.corrugation.config import NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL
from railguard.corrugation.models import ModelSpec
from railguard.corrugation.models_augmented import FittedAugmentedCorrugationModel


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side_i_vibration_rms_median": [0.1, 2.0, 0.2, 0.3, 1.8, 0.2],
            "side_ii_vibration_rms_median": [0.2, 0.1, 2.2, 0.1, 0.2, 1.9],
            "wavelength_valid": [1.0] * 6,
            "tach_transitions": [180.0] * 6,
            "speed_mps": [10.0] * 6,
            "tach_duty": [0.5] * 6,
        }
    )


def test_fractional_duplicate_views_preserve_model_margins() -> None:
    frame = _feature_frame()
    labels = np.asarray(
        [NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL, NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL]
    )
    spec = ModelSpec("all", 0.01, 1.0, False)
    baseline = FittedAugmentedCorrugationModel(spec).fit(frame, labels)

    repeated = frame.loc[frame.index.repeat(9)].reset_index(drop=True)
    repeated_labels = np.repeat(labels, 9)
    weights = np.full(len(repeated), 1.0 / 9.0)
    duplicated = FittedAugmentedCorrugationModel(spec).fit(repeated, repeated_labels, weights)

    baseline_scores = baseline.decision_scores(frame)
    duplicated_scores = duplicated.decision_scores(frame)
    np.testing.assert_allclose(duplicated_scores[0], baseline_scores[0], atol=1e-7)
    np.testing.assert_allclose(duplicated_scores[1], baseline_scores[1], atol=1e-7)


def test_augmented_model_produces_valid_labels() -> None:
    frame = _feature_frame()
    labels = np.asarray(
        [NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL, NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL]
    )
    model = FittedAugmentedCorrugationModel(ModelSpec("all", 0.01, 1.0, False)).fit(frame, labels)

    assert set(model.predict(frame)).issubset({NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL})
