from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railguard.shm.parsing import ShmDataError, load_shm_file


def test_loader_preserves_numeric_signal(simple_signal_csv: Path) -> None:
    signal = load_shm_file(simple_signal_csv, expected_samples=5)
    assert signal.file_id == "signal01.csv"
    assert signal.sample_count == 5
    np.testing.assert_array_equal(signal.values, [0.0, 2.0, 0.0, 2.0, 0.0])
    assert len(signal.sha256) == 64


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("value\n1\n2\n", "header or nonnumeric"),
        ("1,2\n3,4\n", "exactly one column"),
        ("1\nNaN\n", "NaN or infinite"),
    ],
)
def test_loader_rejects_malformed_files(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ShmDataError, match=message):
        load_shm_file(path)


def test_loader_warns_on_unexpected_length(simple_signal_csv: Path) -> None:
    with pytest.warns(UserWarning, match="expected 10"):
        load_shm_file(simple_signal_csv, expected_samples=10)
