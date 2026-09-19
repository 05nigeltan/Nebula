from __future__ import annotations

import numpy as np
import pytest

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.spatial_features import (
    channel_spatial_features,
    distance_from_tachometer,
    resample_to_distance,
)


def _constant_speed_case(
    speed_mps: float,
    wavelength_m: float,
    sample_rate_hz: float = 2_000.0,
    duration_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, CorrugationConfig]:
    sample_count = int(sample_rate_hz * duration_seconds)
    time = np.arange(sample_count) / sample_rate_hz
    distance = speed_mps * time
    transition_rate = 2.0 * 90.0 * speed_mps / (np.pi * 0.85)
    tachometer = (np.floor(time * transition_rate).astype(int) % 2).astype(float)
    signal = np.sin(2.0 * np.pi * distance / wavelength_m)[:, None]
    config = CorrugationConfig(
        expected_samples=sample_count,
        sample_rate_hz=sample_rate_hz,
        spatial_welch_nperseg=1_024,
        spatial_welch_overlap=512,
        spatial_window_nperseg=512,
        spatial_window_overlap=256,
    )
    return tachometer, signal, config


@pytest.mark.parametrize("speed_mps", [5.0, 12.0, 18.0])
def test_spatial_peak_is_stable_across_speed(speed_mps: float) -> None:
    wavelength = 0.08
    tachometer, signal, config = _constant_speed_case(speed_mps, wavelength)
    mapping = distance_from_tachometer(tachometer, config)
    assert mapping is not None
    result = resample_to_distance(tachometer, signal, config)
    assert result is not None
    _, resampled = result
    features = channel_spatial_features(resampled, config)
    assert features["peak_wavelength"][0] == pytest.approx(wavelength, abs=0.008)
    assert features["peak_prominence"][0] > 1.0
    assert features["persistence"][0] > 0.8


def test_spatial_mapping_rejects_insufficient_tachometer_edges() -> None:
    config = CorrugationConfig(expected_samples=100, spatial_min_transitions=20)
    tachometer = np.zeros(100)
    assert distance_from_tachometer(tachometer, config) is None
