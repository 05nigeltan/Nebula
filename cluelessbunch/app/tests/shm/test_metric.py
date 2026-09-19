from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from railguard.shm.metric import score_shm, validate_prediction_frame


def test_exact_competition_metric() -> None:
    truth = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    predicted = np.array([0.15, 0.28, 0.55, 0.68, 0.85])
    metrics = score_shm(truth, predicted)
    expected = np.mean(np.abs(truth - predicted) / truth)
    assert metrics.mape == pytest.approx(expected)
    assert metrics.official_score == pytest.approx(1.0 - expected)


def test_prediction_contract_rejects_nonpositive_values() -> None:
    frame = pd.DataFrame({"file_id": ["a.csv"], "prediction": [0.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        validate_prediction_frame(frame)
