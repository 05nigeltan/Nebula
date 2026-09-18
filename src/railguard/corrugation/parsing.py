"""Strict parsing, channel mapping, manifests, and source identity checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from railguard.corrugation.config import VALID_LABELS

CHANNEL_PATTERN = re.compile(
    r"^(Vibration|Shock) of bearing in position ([1-8]) of car ([1-8])$"
)


class CorrugationDataError(ValueError):
    """Raised when a corrugation file violates the documented contract."""


@dataclass(frozen=True)
class ChannelInfo:
    column_index: int
    signal_type: str
    position: int
    car: int
    side: str


@dataclass(frozen=True)
class CorrugationSignal:
    file_id: str
    tachometer: np.ndarray
    signals: np.ndarray
    channels: tuple[ChannelInfo, ...]
    sha256: str

    @property
    def sample_count(self) -> int:
        return len(self.tachometer)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_channel_headers(columns: list[str]) -> tuple[ChannelInfo, ...]:
    if len(columns) != 128:
        raise CorrugationDataError(f"Expected 128 sensor columns, found {len(columns)}")
    channels: list[ChannelInfo] = []
    observed: set[tuple[int, int, str]] = set()
    for index, column in enumerate(columns):
        match = CHANNEL_PATTERN.fullmatch(str(column))
        if match is None:
            raise CorrugationDataError(f"Unrecognized sensor column: {column!r}")
        signal_type = match.group(1).lower()
        position = int(match.group(2))
        car = int(match.group(3))
        key = (car, position, signal_type)
        if key in observed:
            raise CorrugationDataError(f"Duplicate sensor channel: {column!r}")
        observed.add(key)
        channels.append(
            ChannelInfo(
                column_index=index,
                signal_type=signal_type,
                position=position,
                car=car,
                side="side_i" if position % 2 else "side_ii",
            )
        )
    if len(observed) != 128:
        raise CorrugationDataError("The complete 8-car x 8-position x 2-signal grid is required")
    return tuple(channels)


def load_corrugation_file(
    path: str | Path,
    expected_samples: int | None = 10_000,
) -> CorrugationSignal:
    """Load one finite 129-column recording and derive its channel map from headers."""

    path = Path(path)
    if not path.is_file():
        raise CorrugationDataError(f"Corrugation signal does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise CorrugationDataError(f"Could not parse {path.name}: {exc}") from exc
    if frame.shape[1] != 129:
        raise CorrugationDataError(f"{path.name} must contain exactly 129 columns")
    if expected_samples is not None and len(frame) != expected_samples:
        raise CorrugationDataError(
            f"{path.name} has {len(frame)} rows; expected exactly {expected_samples}"
        )
    if frame.columns[0] != "Rotating speed":
        raise CorrugationDataError("The first column must be 'Rotating speed'")
    channels = _parse_channel_headers(frame.columns[1:].tolist())
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise CorrugationDataError(f"{path.name} contains a nonnumeric value") from exc
    if not np.all(np.isfinite(values)):
        raise CorrugationDataError(f"{path.name} contains NaN or infinite values")
    tach = values[:, 0]
    rounded = np.rint(tach)
    if not np.all(np.isclose(tach, rounded, atol=1e-6)) or not set(np.unique(rounded)).issubset(
        {0.0, 1.0}
    ):
        raise CorrugationDataError(f"{path.name} tachometer must contain only binary 0/1 values")
    return CorrugationSignal(
        file_id=path.name,
        tachometer=rounded.astype(np.float32),
        signals=values[:, 1:],
        channels=channels,
        sha256=sha256_file(path),
    )


def load_training_manifest(train_dir: str | Path, labels_csv: str | Path) -> pd.DataFrame:
    """Return a deterministic one-to-one file/label manifest."""

    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    if not train_dir.is_dir():
        raise CorrugationDataError(f"Training directory does not exist: {train_dir}")
    if not labels_csv.is_file():
        raise CorrugationDataError(f"Label file does not exist: {labels_csv}")
    labels = pd.read_csv(labels_csv, dtype=str)
    if tuple(labels.columns) != ("filename", "label"):
        raise CorrugationDataError("Labels must have exactly the columns filename,label")
    if labels["filename"].duplicated().any():
        raise CorrugationDataError("Labels contain duplicate filenames")
    unknown = sorted(set(labels["label"]).difference(VALID_LABELS))
    if unknown:
        raise CorrugationDataError(f"Unknown corrugation labels: {unknown}")
    actual = sorted(path.name for path in train_dir.glob("*.csv") if path.is_file())
    expected = sorted(labels["filename"].tolist())
    if actual != expected:
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise CorrugationDataError(f"Label/file mismatch; missing={missing}, extra={extra}")
    order = labels["filename"].str.extract(r"(\d+)", expand=False)
    if order.isna().any():
        raise CorrugationDataError("Every training filename must contain a numeric identifier")
    return labels.assign(_order=order.astype(int)).sort_values("_order").drop(columns="_order").reset_index(drop=True)

