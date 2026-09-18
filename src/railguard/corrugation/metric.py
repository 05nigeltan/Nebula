"""Macro-F1 scoring and output-contract validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from railguard.corrugation.config import OUTPUT_COLUMNS, VALID_LABELS
from railguard.corrugation.parsing import CorrugationDataError


@dataclass(frozen=True)
class CorrugationScore:
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "macro_f1": self.macro_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
        }


def score_corrugation(truth: np.ndarray, predicted: np.ndarray) -> CorrugationScore:
    labels = list(VALID_LABELS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        zero_division=0,
    )
    return CorrugationScore(
        macro_f1=float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        per_class_precision=dict(zip(labels, precision.astype(float), strict=True)),
        per_class_recall=dict(zip(labels, recall.astype(float), strict=True)),
        per_class_f1=dict(zip(labels, f1.astype(float), strict=True)),
    )


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise CorrugationDataError(
            f"Corrugation predictions must have columns {OUTPUT_COLUMNS}, got {tuple(frame.columns)}"
        )
    if frame.empty:
        raise CorrugationDataError("Corrugation prediction output cannot be empty")
    if frame["file_id"].isna().any() or frame["file_id"].duplicated().any():
        raise CorrugationDataError("Prediction file IDs must be nonempty and unique")
    if not frame["file_id"].astype(str).str.lower().str.endswith(".csv").all():
        raise CorrugationDataError("Every prediction file_id must include its .csv extension")
    unknown = sorted(set(frame["prediction"]).difference(VALID_LABELS))
    if unknown:
        raise CorrugationDataError(f"Unknown corrugation prediction labels: {unknown}")

