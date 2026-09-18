import numpy as np
import pytest

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.features import extract_features, tachometer_speed
from railguard.corrugation.parsing import load_corrugation_file


def test_tachometer_speed_counts_both_edges() -> None:
    config = CorrugationConfig(expected_samples=8, sample_rate_hz=8.0)
    transitions, speed, duty = tachometer_speed(np.array([0, 0, 1, 1, 0, 0, 1, 1]), config)
    assert transitions == 3
    assert speed == pytest.approx(np.pi * 0.85 * 3 / 180)
    assert duty == 0.5


def test_feature_extraction_is_finite_and_side_symmetric(corrugation_csv) -> None:
    config = CorrugationConfig(
        expected_samples=256,
        sample_rate_hz=256.0,
        welch_nperseg=128,
        welch_overlap=64,
        stft_nperseg=64,
        stft_overlap=32,
    )
    signal = load_corrugation_file(corrugation_csv, expected_samples=256)
    features = extract_features(signal, config)
    first = {name.removeprefix("side_i_") for name in features if name.startswith("side_i_")}
    second = {name.removeprefix("side_ii_") for name in features if name.startswith("side_ii_")}
    assert first == second
    numeric = [value for value in features.values() if isinstance(value, (int, float))]
    assert np.all(np.isfinite(numeric))
