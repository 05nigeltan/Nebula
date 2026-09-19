"""Door CSV/Excel loading and strict timestamp parsing."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from railguard.door.config import NUMERIC_COLUMNS, REQUIRED_COLUMNS, TIME_COLUMN

EXCEL_SUFFIXES = {".xlsx", ".xls"}


class DoorDataError(ValueError):
    """Raised when a Door input violates the expected data contract."""


def parse_door_timestamps(values: pd.Series) -> pd.Series:
    """Parse native strings or genuine spreadsheet datetime cells."""

    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        try:
            parsed = pd.to_datetime(values, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DoorDataError(f"Invalid Door timestamp: {exc}") from exc
        if parsed.isna().any():
            raise DoorDataError("Door timestamps contain missing values")
        return parsed

    raw = values.astype("string")
    parts = raw.str.split("-", expand=True)
    if parts.shape[1] != 7:
        raise DoorDataError("Door timestamps must contain exactly seven hyphen-separated fields")

    try:
        numeric = parts.apply(pd.to_numeric, errors="raise").astype("int64")
        base = pd.to_datetime(
            {
                "year": numeric[0],
                "month": numeric[1],
                "day": numeric[2],
                "hour": numeric[3],
                "minute": numeric[4],
                "second": numeric[5],
            },
            errors="raise",
        )
        parsed = base + pd.to_timedelta(numeric[6], unit="ms")
    except (TypeError, ValueError) as exc:
        raise DoorDataError(f"Invalid Door timestamp: {exc}") from exc

    if parsed.isna().any():
        raise DoorDataError("Door timestamps contain missing values")
    return parsed


def _source_suffix(source: str | Path | Any) -> str:
    name = source if isinstance(source, (str, Path)) else getattr(source, "name", "")
    return Path(str(name)).suffix.lower()


def load_door_data(
    source: str | Path | IO[str] | IO[bytes] | Any,
    *,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load and validate a Door sensor stream from CSV or Excel.

    Excel input reads the selected worksheet. The returned frame preserves the
    original timestamps and adds the private `_parsed_time` column used internally.
    """

    suffix = _source_suffix(source)
    try:
        if suffix in EXCEL_SUFFIXES:
            frame = pd.read_excel(source, sheet_name=sheet_name)
            source_kind = "Excel worksheet"
        elif suffix in {"", ".csv"}:
            frame = pd.read_csv(source)
            source_kind = "CSV"
        else:
            raise DoorDataError(f"Unsupported Door file type {suffix!r}; use .csv, .xlsx, or .xls")
    except Exception as exc:  # pandas exposes multiple parser exception types
        if isinstance(exc, DoorDataError):
            raise
        raise DoorDataError(f"Could not read Door data: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DoorDataError(f"Door {source_kind} is missing required columns: {missing}")
    if frame.empty:
        raise DoorDataError(f"Door {source_kind} contains no readings")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    if frame[TIME_COLUMN].isna().any():
        raise DoorDataError("Door timestamps contain missing values")

    for column in NUMERIC_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DoorDataError(f"Column {column!r} contains a nonnumeric value") from exc
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DoorDataError(f"Column {column!r} contains a non-finite value")

    frame["_parsed_time"] = parse_door_timestamps(frame[TIME_COLUMN])
    differences = frame["_parsed_time"].diff().iloc[1:]
    if (differences <= pd.Timedelta(0)).any():
        raise DoorDataError("Door timestamps must be strictly increasing with no duplicates")
    return frame


def load_door_csv(source: str | Path | IO[str] | IO[bytes] | Any) -> pd.DataFrame:
    """Backward-compatible wrapper for existing CSV training code."""

    return load_door_data(source)


def parse_scalar_timestamp(value: str) -> pd.Timestamp:
    """Parse one native or ISO-compatible timestamp for scoring."""

    native = pd.Series([value], dtype="string")
    if str(value).count("-") == 6:
        return parse_door_timestamps(native).iloc[0]
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DoorDataError(f"Invalid timestamp {value!r}") from exc
    return pd.Timestamp(parsed)
