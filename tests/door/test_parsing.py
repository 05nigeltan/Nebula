from __future__ import annotations

import pandas as pd
import pytest

from railguard.door.parsing import (
    DoorDataError,
    load_door_csv,
    load_door_data,
    parse_door_timestamps,
)


def test_native_timestamp_parser_preserves_milliseconds() -> None:
    parsed = parse_door_timestamps(pd.Series(["2023-7-5-1-2-3-47"]))
    assert parsed.iloc[0] == pd.Timestamp("2023-07-05 01:02:03.047")


def test_timestamp_parser_accepts_excel_datetime_cells() -> None:
    values = pd.Series(pd.to_datetime(["2023-07-05 01:02:03.047"]))
    parsed = parse_door_timestamps(values)
    assert parsed.iloc[0] == pd.Timestamp("2023-07-05 01:02:03.047")


def test_loader_rejects_duplicate_timestamps(two_cycle_csv) -> None:
    frame = pd.read_csv(two_cycle_csv)
    frame.loc[1, "Datetime"] = frame.loc[0, "Datetime"]
    frame.to_csv(two_cycle_csv, index=False)
    with pytest.raises(DoorDataError, match="strictly increasing"):
        load_door_csv(two_cycle_csv)


def test_loader_reads_selected_excel_worksheet(two_cycle_excel) -> None:
    frame = load_door_data(two_cycle_excel, sheet_name="Door readings")
    assert len(frame) == 6
    assert frame["_parsed_time"].is_monotonic_increasing
