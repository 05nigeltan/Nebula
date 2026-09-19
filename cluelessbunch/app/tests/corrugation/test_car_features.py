from dataclasses import replace

import numpy as np
import pandas as pd

from railguard.corrugation.car_features import car_contrast_features
from railguard.corrugation.models import feature_bases
from railguard.corrugation.parsing import load_corrugation_file


def test_car_features_swap_sides_and_survive_common_gain(corrugation_csv):
    signal = load_corrugation_file(corrugation_csv, 256)
    original = car_contrast_features(signal)
    swapped = replace(
        signal,
        channels=tuple(
            replace(c, side="side_ii" if c.side == "side_i" else "side_i") for c in signal.channels
        ),
    )
    reversed_features = car_contrast_features(swapped)
    scaled = car_contrast_features(replace(signal, signals=signal.signals * 8))
    for name, value in original.items():
        other = (
            name.replace("side_i_", "side_ii_", 1)
            if name.startswith("side_i_")
            else name.replace("side_ii_", "side_i_", 1)
        )
        assert value == reversed_features[other]
        np.testing.assert_allclose(value, scaled[name], atol=1e-7)
    assert len(original) == 18
    frame = pd.DataFrame([original])
    assert len(feature_bases(frame, "time")) == 9
    assert len(feature_bases(frame, "compact")) == 9


def test_car_features_distinguish_local_from_repeated_anomalies(corrugation_csv):
    signal = load_corrugation_file(corrugation_csv, 256)
    wave = np.sin(np.arange(256) * 2 * np.pi / 32)
    local = np.column_stack(
        [wave * (3 if c.car == 1 and c.side == "side_i" else 1) for c in signal.channels]
    )
    repeated = np.column_stack([wave * (3 if c.side == "side_i" else 1) for c in signal.channels])
    local_features = car_contrast_features(replace(signal, signals=local))
    repeated_features = car_contrast_features(replace(signal, signals=repeated))
    assert local_features["side_i_car_rms_joint_positive_fraction"] == 1 / 8
    assert repeated_features["side_i_car_rms_joint_positive_fraction"] == 1
    silent = car_contrast_features(replace(signal, signals=np.zeros_like(local)))
    assert set(silent.values()) == {0.0}


def test_car_renumbering_preserves_aggregate_features(corrugation_csv):
    signal = load_corrugation_file(corrugation_csv, 256)
    renumbered = replace(signal, channels=tuple(replace(c, car=9 - c.car) for c in signal.channels))
    a, b = car_contrast_features(signal), car_contrast_features(renumbered)
    np.testing.assert_allclose(list(a.values()), list(b.values()), atol=1e-14)
