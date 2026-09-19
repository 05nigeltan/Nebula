"""Exact SHM competition metric and useful guardrail diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUTPUT_COLUMNS = ("file_id", "prediction")


@dataclass(frozen=True)
class ShmMetrics:
    mape: float
    official_score: float
    median_ape: float
    p90_ape: float
    max_ape: float
    signed_bias: float
    spearman: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def score_shm(y_true: np.ndarray, y_pred: np.ndarray) -> ShmMetrics:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("SHM truth and predictions must be same-length one-dimensional arrays")
    if np.any(truth <= 0) or not np.all(np.isfinite(truth)):
        raise ValueError("SHM truth must be finite and strictly positive")
    if np.any(predicted <= 0) or not np.all(np.isfinite(predicted)):
        raise ValueError("SHM predictions must be finite and strictly positive")
    relative = (predicted - truth) / truth
    absolute = np.abs(relative)
    correlation = spearmanr(truth, predicted).statistic if len(truth) > 1 else np.nan
    return ShmMetrics(
        mape=float(np.mean(absolute)),
        official_score=float(max(0.0, 1.0 - np.mean(absolute))),
        median_ape=float(np.median(absolute)),
        p90_ape=float(np.quantile(absolute, 0.90)),
        max_ape=float(np.max(absolute)),
        signed_bias=float(np.mean(relative)),
        spearman=float(correlation),
    )


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Validate the exact competition-facing SHM output contract."""

    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            f"SHM predictions must have columns {OUTPUT_COLUMNS}, got {tuple(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("SHM predictions cannot be empty")
    if frame["file_id"].isna().any() or frame["file_id"].astype(str).duplicated().any():
        raise ValueError("SHM file_id values must be present and unique")
    predictions = pd.to_numeric(frame["prediction"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(predictions)) or np.any(predictions <= 0):
        raise ValueError("SHM predictions must be finite and strictly positive")
