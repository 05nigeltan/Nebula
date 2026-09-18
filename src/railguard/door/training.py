"""Leakage-resistant model comparison, training, and artifact creation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from railguard.door.config import ABNORMAL_LABEL, NORMAL_LABEL, DoorConfig
from railguard.door.metric import DoorScore, score_door_segments
from railguard.door.models import OperationThresholdClassifier, build_model, model_scores
from railguard.door.parsing import DoorDataError, load_door_csv
from railguard.door.segmentation import DoorSegment, SegmentationDiagnostics, segment_stream


@dataclass(frozen=True)
class CandidateResult:
    name: str
    official_score: float
    accuracy: float
    abnormal_precision: float
    abnormal_recall: float
    open_accuracy: float
    close_accuracy: float
    fold_choices: list[dict[str, Any]]


CANDIDATE_PARAMETERS: dict[str, tuple[float | None, ...]] = {
    "logistic_minimal": (0.01, 0.1, 1.0, 10.0),
    "logistic_physics": (0.01, 0.1, 1.0, 10.0),
    "linear_svm": (0.01, 0.1, 1.0, 10.0),
    "shallow_boosting": (None,),
}

COMPLEXITY_ORDER = {
    "threshold_baseline": 0,
    "logistic_minimal": 1,
    "logistic_physics": 2,
    "linear_svm": 3,
    "shallow_boosting": 4,
}


def _subset(items: list[DoorSegment], indices: np.ndarray) -> list[DoorSegment]:
    return [items[int(index)] for index in indices]


def contiguous_folds(indices: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create held-out contiguous blocks while preserving original stream order."""

    ordered = np.sort(np.asarray(indices, dtype=int))
    blocks = [block for block in np.array_split(ordered, folds) if len(block)]
    result = []
    for block in blocks:
        validation = np.asarray(block, dtype=int)
        training = np.asarray([value for value in ordered if value not in set(block)], dtype=int)
        result.append((training, validation))
    return result


def load_labelled_cycles(
    train_csv: str | Path,
    labels_csv: str | Path,
    config: DoorConfig,
) -> tuple[list[DoorSegment], np.ndarray, pd.DataFrame, SegmentationDiagnostics]:
    frame = load_door_csv(train_csv)
    segments, diagnostics = segment_stream(frame, gap_threshold_ms=config.gap_threshold_ms)
    labels = pd.read_csv(labels_csv, dtype=str)
    required = {"start_time", "end_time", "operation", "status"}
    if missing := required.difference(labels.columns):
        raise DoorDataError(f"Door labels are missing columns: {sorted(missing)}")
    if len(labels) != len(segments):
        raise DoorDataError(
            f"Detected {len(segments)} cycles but label file contains {len(labels)} rows"
        )

    targets = []
    for segment, label in zip(segments, labels.itertuples(index=False), strict=True):
        if segment.start_time != label.start_time or segment.end_time != label.end_time:
            raise DoorDataError(
                f"Segment {segment.segment_index} does not exactly match labelled boundaries"
            )
        if segment.operation != label.operation:
            raise DoorDataError(
                f"Segment {segment.segment_index} operation is {segment.operation}, "
                f"expected {label.operation}"
            )
        if label.status == ABNORMAL_LABEL:
            targets.append(1)
        elif label.status == NORMAL_LABEL:
            targets.append(0)
        else:
            raise DoorDataError(f"Unknown training label {label.status!r}")
    truth = labels.loc[:, ["start_time", "end_time", "status"]].copy()
    return segments, np.asarray(targets, dtype=int), truth, diagnostics


def choose_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    """Choose score threshold by accuracy, then abnormal recall, then centrality."""

    order = np.unique(np.asarray(scores, dtype=float))
    if len(order) == 1:
        candidates = np.array([order[0]])
    else:
        candidates = np.concatenate(
            ([np.nextafter(order[0], -np.inf)], (order[:-1] + order[1:]) / 2.0, [order[-1]])
        )
    best_key: tuple[float, float, float] | None = None
    best = (float(candidates[0]), 0.0, 0.0)
    centre = float(np.median(scores))
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        accuracy = float(np.mean(predicted == targets))
        true_positive = int(np.sum((predicted == 1) & (targets == 1)))
        false_negative = int(np.sum((predicted == 0) & (targets == 1)))
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        key = (accuracy, recall, -abs(float(threshold) - centre))
        if best_key is None or key > best_key:
            best_key = key
            best = (float(threshold), accuracy, recall)
    return best


def _cross_fitted_scores(
    name: str,
    parameter: float | None,
    segments: list[DoorSegment],
    targets: np.ndarray,
    indices: np.ndarray,
    folds: int,
    config: DoorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    collected_scores: list[float] = []
    collected_targets: list[int] = []
    for train_index, validation_index in contiguous_folds(indices, folds):
        model = build_model(name, parameter, config.template_points)
        model.fit(_subset(segments, train_index), targets[train_index])
        collected_scores.extend(model_scores(model, _subset(segments, validation_index)))
        collected_targets.extend(targets[validation_index])
    return np.asarray(collected_scores), np.asarray(collected_targets, dtype=int)


def _select_parameter(
    name: str,
    segments: list[DoorSegment],
    targets: np.ndarray,
    training_indices: np.ndarray,
    config: DoorConfig,
) -> tuple[float | None, float, float]:
    best_key: tuple[float, float, float] | None = None
    best_choice: tuple[float | None, float, float] | None = None
    folds = min(config.inner_folds, len(training_indices))
    for parameter in CANDIDATE_PARAMETERS[name]:
        scores, inner_targets = _cross_fitted_scores(
            name, parameter, segments, targets, training_indices, folds, config
        )
        threshold, accuracy, recall = choose_threshold(scores, inner_targets)
        parameter_penalty = 0.0 if parameter is None else -abs(np.log10(float(parameter))) * 1e-9
        key = (accuracy, recall, parameter_penalty)
        if best_key is None or key > best_key:
            best_key = key
            best_choice = (parameter, threshold, accuracy)
    assert best_choice is not None
    return best_choice


def _prediction_frame(segments: list[DoorSegment], predicted: np.ndarray) -> pd.DataFrame:
    labels = np.where(predicted == 1, ABNORMAL_LABEL, NORMAL_LABEL)
    return pd.DataFrame(
        {
            "start_time": [segment.start_time for segment in segments],
            "end_time": [segment.end_time for segment in segments],
            "prediction": labels,
        }
    )


def _summarise_candidate(
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    segments: list[DoorSegment],
    truth: pd.DataFrame,
    fold_choices: list[dict[str, Any]],
) -> CandidateResult:
    prediction_frame = _prediction_frame(segments, predictions)
    official: DoorScore = score_door_segments(truth, prediction_frame)
    true_positive = int(np.sum((predictions == 1) & (targets == 1)))
    false_positive = int(np.sum((predictions == 1) & (targets == 0)))
    false_negative = int(np.sum((predictions == 0) & (targets == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    operations = np.array([segment.operation for segment in segments])
    return CandidateResult(
        name=name,
        official_score=official.score,
        accuracy=float(np.mean(predictions == targets)),
        abnormal_precision=precision,
        abnormal_recall=recall,
        open_accuracy=float(np.mean(predictions[operations == "Open"] == targets[operations == "Open"])),
        close_accuracy=float(
            np.mean(predictions[operations == "Close"] == targets[operations == "Close"])
        ),
        fold_choices=fold_choices,
    )


def evaluate_candidates(
    segments: list[DoorSegment],
    targets: np.ndarray,
    truth: pd.DataFrame,
    config: DoorConfig,
) -> tuple[list[CandidateResult], dict[str, np.ndarray]]:
    indices = np.arange(len(segments))
    outer = contiguous_folds(indices, config.outer_folds)
    all_predictions: dict[str, np.ndarray] = {}
    results: list[CandidateResult] = []

    baseline_predictions = np.zeros(len(segments), dtype=int)
    baseline_choices = []
    for train_index, validation_index in outer:
        model = OperationThresholdClassifier().fit(_subset(segments, train_index), targets[train_index])
        baseline_predictions[validation_index] = model.predict(_subset(segments, validation_index))
        baseline_choices.append({"thresholds": model.thresholds_, "directions": model.directions_})
    all_predictions["threshold_baseline"] = baseline_predictions
    results.append(
        _summarise_candidate(
            "threshold_baseline",
            baseline_predictions,
            targets,
            segments,
            truth,
            baseline_choices,
        )
    )

    for name in CANDIDATE_PARAMETERS:
        predictions = np.zeros(len(segments), dtype=int)
        choices: list[dict[str, Any]] = []
        for train_index, validation_index in outer:
            parameter, threshold, inner_accuracy = _select_parameter(
                name, segments, targets, train_index, config
            )
            model = build_model(name, parameter, config.template_points)
            model.fit(_subset(segments, train_index), targets[train_index])
            scores = model_scores(model, _subset(segments, validation_index))
            predictions[validation_index] = (scores >= threshold).astype(int)
            choices.append(
                {
                    "parameter": parameter,
                    "threshold": threshold,
                    "inner_accuracy": inner_accuracy,
                }
            )
        all_predictions[name] = predictions
        results.append(
            _summarise_candidate(name, predictions, targets, segments, truth, choices)
        )
    return results, all_predictions


def _fit_final_model(
    name: str,
    segments: list[DoorSegment],
    targets: np.ndarray,
    config: DoorConfig,
) -> tuple[BaseEstimator, float, float | None]:
    indices = np.arange(len(segments))
    if name == "threshold_baseline":
        model = OperationThresholdClassifier().fit(segments, targets)
        return model, 0.0, None
    parameter, threshold, _ = _select_parameter(name, segments, targets, indices, config)
    model = build_model(name, parameter, config.template_points)
    model.fit(segments, targets)
    return model, threshold, parameter


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_and_save(
    train_csv: str | Path,
    labels_csv: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    config: DoorConfig | None = None,
) -> dict[str, Any]:
    """Evaluate bounded candidates, select one, and save a reproducible artifact."""

    config = config or DoorConfig()
    segments, targets, truth, diagnostics = load_labelled_cycles(train_csv, labels_csv, config)
    results, predictions = evaluate_candidates(segments, targets, truth, config)
    acceptable = [result for result in results if result.abnormal_recall >= 0.90]
    if not acceptable:
        acceptable = [result for result in results if result.name == "threshold_baseline"]
    selected = min(
        acceptable,
        key=lambda result: (-result.official_score, COMPLEXITY_ORDER[result.name]),
    )
    model, threshold, parameter = _fit_final_model(selected.name, segments, targets, config)

    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.joblib"
    metadata = {
        "model_name": selected.name,
        "score_threshold": threshold,
        "parameter": parameter,
        "config": asdict(config),
        "train_sha256": _sha256(train_csv),
        "labels_sha256": _sha256(labels_csv),
        "training_cycles": len(segments),
        "class_counts": {
            NORMAL_LABEL: int(np.sum(targets == 0)),
            ABNORMAL_LABEL: int(np.sum(targets == 1)),
        },
        "segmentation": asdict(diagnostics),
        "candidate_results": [asdict(result) for result in results],
    }
    joblib.dump(
        {
            "model": model,
            "model_name": selected.name,
            "score_threshold": threshold,
            "config": asdict(config),
            "metadata": metadata,
        },
        artifact_path,
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    (report_dir / "training_report.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    selected_frame = _prediction_frame(segments, predictions[selected.name])
    selected_frame.insert(0, "segment_id", np.arange(1, len(selected_frame) + 1))
    selected_frame["truth"] = np.where(targets == 1, ABNORMAL_LABEL, NORMAL_LABEL)
    selected_frame.to_csv(report_dir / "out_of_fold_predictions.csv", index=False)
    return {"artifact_path": str(artifact_path), **metadata}
