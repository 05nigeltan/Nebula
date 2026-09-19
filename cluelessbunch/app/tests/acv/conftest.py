from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def acv_workbook(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    sample_count = 120
    frame: dict[str, object] = {
        "Car model": ["Synthetic"] * sample_count,
        "Train number": ["T01"] * sample_count,
        "Time": pd.date_range("2025-01-01", periods=sample_count, freq="30s"),
    }
    for car in ("01", "02", "03", "04"):
        fault_offset = 4.0 if car == "03" else 0.0
        frame[f"Car {car} - Indoor Average Temperature"] = (
            23.0 + fault_offset + rng.normal(0.0, 0.08, sample_count)
        )
        frame[f"Car {car} - Outdoor Average Temperature"] = np.full(sample_count, 31.0)
        frame[f"Car {car} - ACV Control Temperature (Cooling)"] = np.full(sample_count, 22.0)
        frame[f"Car {car} - ACV Running Mode"] = ["Cooling"] * sample_count
        frame[f"Car {car} - ACV Information Valid"] = ["Valid"] * sample_count
    path = tmp_path / "synthetic_acv.xlsx"
    pd.DataFrame(frame).to_excel(path, index=False)
    return path
