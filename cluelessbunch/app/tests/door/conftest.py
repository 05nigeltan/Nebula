from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from railguard.door.config import REQUIRED_COLUMNS
from railguard.project_paths import participant_root

ROOT = Path(__file__).resolve().parents[2]
DOOR_DATA = participant_root(ROOT) / "02_Datasets" / "Door"


@pytest.fixture
def two_cycle_csv(tmp_path: Path) -> Path:
    rows = []
    timestamps = [
        "2023-7-5-0-0-0-0",
        "2023-7-5-0-0-0-20",
        "2023-7-5-0-0-0-40",
        "2023-7-5-0-0-2-0",
        "2023-7-5-0-0-2-20",
        "2023-7-5-0-0-2-40",
    ]
    for index, timestamp in enumerate(timestamps):
        second_cycle = index >= 3
        values = {column: 0 for column in REQUIRED_COLUMNS}
        values["Datetime"] = timestamp
        values["Motor current(mA)"] = 100 + index
        values["Motor Voltage(10mV)"] = 50
        values["Motor electrodynamic force"] = 5
        values["Open command"] = 0 if second_cycle else 1
        values["Close command"] = 1 if second_cycle else 0
        values["Door leaf position"] = (5 - index) if second_cycle else index
        rows.append(values)
    path = tmp_path / "stream.csv"
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def two_cycle_excel(two_cycle_csv: Path, tmp_path: Path) -> Path:
    path = tmp_path / "stream.xlsx"
    pd.read_csv(two_cycle_csv).to_excel(path, sheet_name="Door readings", index=False)
    return path
