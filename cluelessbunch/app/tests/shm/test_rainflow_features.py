from __future__ import annotations

import numpy as np

from railguard.shm.config import ShmConfig
from railguard.shm.parsing import ShmSignal
from railguard.shm.rainflow_features import extract_features


def test_repeated_triangle_has_two_weighted_cycles() -> None:
    signal = ShmSignal(
        file_id="triangle.csv",
        values=np.array([0.0, 2.0, 0.0, 2.0, 0.0]),
        sha256="0" * 64,
    )
    features = extract_features(signal, ShmConfig(expected_samples=5))
    assert features["rf_weighted_cycle_count"] == 2.0
    assert features["rf_half_cycle_entries"] == 4
    assert np.isclose(features["log_b_m_5"], np.log(2.0))
    assert np.isclose(
        sum(features[f"rf_bin_{index}_damage_share"] for index in range(8)),
        1.0,
    )
    assert np.isclose(
        sum(features[f"rf_quarter_{index}_damage_share"] for index in range(4)),
        1.0,
    )


def test_feature_extraction_is_deterministic() -> None:
    signal = ShmSignal(
        file_id="deterministic.csv",
        values=np.sin(np.linspace(0.0, 20.0 * np.pi, 1000)),
        sha256="1" * 64,
    )
    config = ShmConfig(expected_samples=1000)
    assert extract_features(signal, config) == extract_features(signal, config)
