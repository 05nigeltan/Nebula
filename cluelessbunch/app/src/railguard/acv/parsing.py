"""Schema-aware parsing and manifest validation for ACV Excel workbooks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CAR_COLUMN_PATTERN = re.compile(r"^Car\s+(\d{2})\s*-\s*(.+)$")
IDENTIFIER_COLUMNS = ("Car model", "Train number", "Time")


class AcvDataError(ValueError):
    """Raised when an ACV workbook violates the documented data contract."""


@dataclass(frozen=True)
class AcvCase:
    file_id: str
    frame: pd.DataFrame
    timestamps: pd.Series
    car_columns: dict[str, dict[str, str]]
    sheet_name: str
    sha256: str | None = None

    @property
    def car_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.car_columns, key=lambda value: int(value)))

    @property
    def sample_count(self) -> int:
        return len(self.frame)

    def subset(self, indices: Any) -> AcvCase:
        frame = self.frame.iloc[indices].reset_index(drop=True)
        timestamps = self.timestamps.iloc[indices].reset_index(drop=True)
        return AcvCase(
            file_id=self.file_id,
            frame=frame,
            timestamps=timestamps,
            car_columns=self.car_columns,
            sheet_name=self.sheet_name,
            sha256=self.sha256,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_name(source: str | Path | Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    return Path(str(name)).name if name else "uploaded_acv_case.xlsx"


def _car_column_map(columns: list[object]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw_column in columns:
        column = str(raw_column).strip()
        match = CAR_COLUMN_PATTERN.fullmatch(column)
        if match is None:
            continue
        car, parameter = match.groups()
        parameter = parameter.strip()
        if parameter in result.setdefault(car, {}):
            raise AcvDataError(f"Duplicate parameter for Car {car}: {parameter}")
        result[car][parameter] = column
    if len(result) < 2:
        raise AcvDataError("An ACV workbook must contain telemetry for at least two cars")
    return result


def load_acv_case(source: str | Path | Any, sheet_name: str | int = 0) -> AcvCase:
    """Load one ACV workbook and derive its car/parameter mapping from the headers."""

    file_id = _source_name(source)
    if Path(file_id).suffix.lower() != ".xlsx":
        raise AcvDataError("ACV input must be an .xlsx workbook")
    try:
        with pd.ExcelFile(source) as workbook:
            if isinstance(sheet_name, int):
                if sheet_name < 0 or sheet_name >= len(workbook.sheet_names):
                    raise AcvDataError(f"Worksheet index {sheet_name} is out of range")
                selected_sheet = workbook.sheet_names[sheet_name]
            else:
                if sheet_name not in workbook.sheet_names:
                    raise AcvDataError(f"Worksheet does not exist: {sheet_name}")
                selected_sheet = sheet_name
            frame = pd.read_excel(workbook, sheet_name=selected_sheet)
    except AcvDataError:
        raise
    except (OSError, ValueError, ImportError, TypeError) as exc:
        raise AcvDataError(f"Could not parse {file_id}: {exc}") from exc

    if frame.empty:
        raise AcvDataError(f"{file_id} contains no data rows")
    if frame.shape[1] < 7:
        raise AcvDataError(f"{file_id} does not contain enough ACV telemetry columns")
    if tuple(map(str, frame.columns[:3])) != IDENTIFIER_COLUMNS:
        raise AcvDataError(
            f"The first three columns must be {IDENTIFIER_COLUMNS}, got "
            f"{tuple(map(str, frame.columns[:3]))}"
        )
    timestamps = pd.to_datetime(frame["Time"], errors="coerce")
    if timestamps.isna().any():
        raise AcvDataError(f"{file_id} contains an invalid timestamp")
    if not timestamps.is_monotonic_increasing:
        raise AcvDataError(f"{file_id} timestamps must be in increasing order")
    car_columns = _car_column_map(frame.columns[3:].tolist())
    digest = sha256_file(source) if isinstance(source, (str, Path)) else None
    return AcvCase(
        file_id=file_id,
        frame=frame,
        timestamps=timestamps,
        car_columns=car_columns,
        sheet_name=str(selected_sheet),
        sha256=digest,
    )


def load_training_manifest(train_dir: str | Path, labels_csv: str | Path) -> pd.DataFrame:
    """Return a deterministic one-to-one ACV workbook and faulty-car manifest."""

    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    if not train_dir.is_dir():
        raise AcvDataError(f"Training directory does not exist: {train_dir}")
    if not labels_csv.is_file():
        raise AcvDataError(f"Label file does not exist: {labels_csv}")
    labels = pd.read_csv(labels_csv, dtype=str)
    if tuple(labels.columns) != ("filename", "faulty_car"):
        raise AcvDataError("ACV labels must have exactly the columns filename,faulty_car")
    if labels.isna().any().any() or labels["filename"].duplicated().any():
        raise AcvDataError("ACV labels must be complete and filenames must be unique")
    if not labels["faulty_car"].str.fullmatch(r"\d{2}").all():
        raise AcvDataError("Every faulty_car must be a two-digit car identifier")
    actual = sorted(path.name for path in train_dir.glob("*.xlsx") if path.is_file())
    expected = sorted(labels["filename"].tolist())
    if actual != expected:
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise AcvDataError(f"Label/file mismatch; missing={missing}, extra={extra}")
    order = labels["filename"].str.extract(r"(\d+)", expand=False)
    if order.isna().any():
        raise AcvDataError("Every ACV training filename must contain a numeric identifier")
    return (
        labels.assign(_order=order.astype(int))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
