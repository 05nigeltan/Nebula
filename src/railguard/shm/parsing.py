"""Strict parsing and identity checks for SHM signal files and labels."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class ShmDataError(ValueError):
    """Raised when SHM data violates the documented input contract."""


@dataclass(frozen=True)
class ShmSignal:
    file_id: str
    values: np.ndarray
    sha256: str

    @property
    def sample_count(self) -> int:
        return len(self.values)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shm_file(path: str | Path, expected_samples: int | None = None) -> ShmSignal:
    """Load one headerless, single-column, finite numeric stress signal."""

    path = Path(path)
    if not path.is_file():
        raise ShmDataError(f"SHM signal does not exist: {path}")
    try:
        frame = pd.read_csv(path, header=None)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ShmDataError(f"Could not parse {path.name}: {exc}") from exc
    if frame.empty:
        raise ShmDataError(f"{path.name} is empty")
    if frame.shape[1] != 1:
        raise ShmDataError(f"{path.name} must contain exactly one column")
    try:
        values = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ShmDataError(f"{path.name} contains a header or nonnumeric value") from exc
    if not np.all(np.isfinite(values)):
        raise ShmDataError(f"{path.name} contains NaN or infinite values")
    if expected_samples is not None and len(values) != expected_samples:
        warnings.warn(
            f"{path.name} has {len(values)} samples; expected {expected_samples}",
            stacklevel=2,
        )
    return ShmSignal(path.name, values, sha256_file(path))


def load_training_manifest(
    train_dir: str | Path,
    labels_csv: str | Path,
) -> pd.DataFrame:
    """Return a deterministic one-to-one file/positive-label manifest."""

    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    if not train_dir.is_dir():
        raise ShmDataError(f"Training directory does not exist: {train_dir}")
    if not labels_csv.is_file():
        raise ShmDataError(f"Label file does not exist: {labels_csv}")
    labels = pd.read_csv(labels_csv)
    if tuple(labels.columns) != ("filename", "damage"):
        raise ShmDataError("SHM labels must have exactly the columns filename,damage")
    if labels["filename"].duplicated().any():
        raise ShmDataError("SHM labels contain duplicate filenames")
    labels["filename"] = labels["filename"].astype(str)
    labels["damage"] = pd.to_numeric(labels["damage"], errors="raise")
    if not np.all(np.isfinite(labels["damage"])) or np.any(labels["damage"] <= 0):
        raise ShmDataError("Every SHM damage label must be finite and strictly positive")
    actual = sorted(path.name for path in train_dir.glob("*.csv") if path.is_file())
    expected = sorted(labels["filename"].tolist())
    if actual != expected:
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ShmDataError(f"Label/file mismatch; missing={missing}, extra={extra}")
    natural_order = labels["filename"].str.extract(r"(\d+)", expand=False).astype(int)
    return labels.assign(_order=natural_order).sort_values("_order").drop(columns="_order").reset_index(drop=True)
