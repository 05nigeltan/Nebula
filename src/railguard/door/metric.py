"""Exact implementation of the official Door IoU-weighted F1 metric."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from railguard.door.config import OUTPUT_COLUMNS, VALID_LABELS
from railguard.door.parsing import DoorDataError, parse_scalar_timestamp


@dataclass(frozen=True)
class DoorScore:
    score: float
    soft_precision: float
    soft_recall: float
    matched_iou_sum: float
    matches: int
    true_segments: int
    predicted_segments: int


def _prepare_segments(frame: pd.DataFrame, label_column: str) -> list[tuple[int, int, str]]:
    required = {"start_time", "end_time", label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise DoorDataError(f"Segment table is missing columns: {sorted(missing)}")

    result: list[tuple[int, int, str]] = []
    for row in frame.itertuples(index=False):
        start = parse_scalar_timestamp(str(row.start_time))
        end = parse_scalar_timestamp(str(row.end_time))
        label = str(getattr(row, label_column))
        if label not in VALID_LABELS:
            raise DoorDataError(f"Unknown Door label {label!r}")
        if end <= start:
            raise DoorDataError("Every Door segment must have end_time after start_time")
        result.append((start.value, end.value, label))
    return result


def score_door_segments(truth: pd.DataFrame, predictions: pd.DataFrame) -> DoorScore:
    """Score predictions using same-label, greedy, one-to-one temporal IoU matching."""

    truth_label = "status" if "status" in truth.columns else "prediction"
    true_segments = _prepare_segments(truth, truth_label)
    predicted_segments = _prepare_segments(predictions, "prediction")

    candidates: list[tuple[float, int, int]] = []
    for true_index, (true_start, true_end, true_label) in enumerate(true_segments):
        true_duration = true_end - true_start
        for pred_index, (pred_start, pred_end, pred_label) in enumerate(predicted_segments):
            if true_label != pred_label:
                continue
            intersection = max(0, min(true_end, pred_end) - max(true_start, pred_start))
            if intersection <= 0:
                continue
            union = true_duration + (pred_end - pred_start) - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > 0:
                candidates.append((iou, true_index, pred_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_true: set[int] = set()
    used_predicted: set[int] = set()
    matched_iou_sum = 0.0
    matches = 0
    for iou, true_index, pred_index in candidates:
        if true_index in used_true or pred_index in used_predicted:
            continue
        used_true.add(true_index)
        used_predicted.add(pred_index)
        matched_iou_sum += iou
        matches += 1

    recall = matched_iou_sum / len(true_segments) if true_segments else 0.0
    precision = matched_iou_sum / len(predicted_segments) if predicted_segments else 0.0
    score = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return DoorScore(
        score=score,
        soft_precision=precision,
        soft_recall=recall,
        matched_iou_sum=matched_iou_sum,
        matches=matches,
        true_segments=len(true_segments),
        predicted_segments=len(predicted_segments),
    )


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    """Validate the exact submission-facing output contract."""

    if tuple(predictions.columns) != OUTPUT_COLUMNS:
        raise DoorDataError(
            f"Door predictions must have columns {OUTPUT_COLUMNS}, got {tuple(predictions.columns)}"
        )
    _prepare_segments(predictions, "prediction")
