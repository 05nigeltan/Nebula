from __future__ import annotations

import numpy as np
import pandas as pd

from railguard.corrugation.augmentation import (
    extract_jackknife_views,
    full_features_from_views,
    select_views,
    validate_view_groups,
)
from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.features import extract_features
from railguard.corrugation.parsing import load_corrugation_file


def _small_config() -> CorrugationConfig:
    return CorrugationConfig(
        expected_samples=256,
        sample_rate_hz=256.0,
        welch_nperseg=128,
        welch_overlap=64,
        stft_nperseg=64,
        stft_overlap=32,
    )


def test_full_sensor_view_matches_public_extractor(corrugation_csv) -> None:
    config = _small_config()
    signal = load_corrugation_file(corrugation_csv, config.expected_samples)
    expected = extract_features(signal, config)
    views = pd.DataFrame(extract_jackknife_views(signal, config))

    validate_view_groups(views)
    actual = full_features_from_views(views).iloc[0].to_dict()

    assert set(actual) == set(expected)
    for name, value in expected.items():
        if isinstance(value, str):
            assert actual[name] == value
        else:
            assert actual[name] == value


def test_jackknife_views_are_complete_and_weight_conserving(corrugation_csv) -> None:
    config = _small_config()
    signal = load_corrugation_file(corrugation_csv, config.expected_samples)
    views = pd.DataFrame(extract_jackknife_views(signal, config))

    assert len(views) == 9
    assert set(views["omitted_car"]) == set(range(9))
    assert np.isclose(views["view_weight"].sum(), 1.0)
    assert views.filter(regex=r"^side_(i|ii)_").notna().all().all()


def test_select_views_never_adds_an_unrequested_source(corrugation_csv) -> None:
    config = _small_config()
    signal = load_corrugation_file(corrugation_csv, config.expected_samples)
    first = pd.DataFrame(extract_jackknife_views(signal, config))
    second = first.assign(
        file_id="second.csv",
        source_file_id="second.csv",
        sha256="different-hash",
    )
    combined = pd.concat([first, second], ignore_index=True)

    selected = select_views(combined, [signal.file_id])

    assert set(selected["source_file_id"]) == {signal.file_id}
    assert len(selected) == 9
