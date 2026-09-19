from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def corrugation_csv(tmp_path: Path) -> Path:
    samples = 256
    time = np.arange(samples) / samples
    columns: dict[str, np.ndarray] = {"Rotating speed": ((np.arange(samples) // 8) % 2)}
    for car in range(1, 9):
        for position in range(1, 9):
            phase = 0.1 * car + 0.2 * position
            columns[f"Vibration of bearing in position {position} of car {car}"] = np.sin(
                2 * np.pi * (10 + position) * time + phase
            )
            columns[f"Shock of bearing in position {position} of car {car}"] = np.cos(
                2 * np.pi * (20 + car) * time + phase
            )
    path = tmp_path / "sample.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    return path
