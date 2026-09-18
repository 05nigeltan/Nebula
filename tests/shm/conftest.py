from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def simple_signal_csv(tmp_path: Path) -> Path:
    path = tmp_path / "signal01.csv"
    np.savetxt(path, np.array([0.0, 2.0, 0.0, 2.0, 0.0]), delimiter=",")
    return path
