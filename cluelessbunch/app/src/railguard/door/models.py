"""Door model definitions and score helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from railguard.door.config import CURRENT_COLUMN
from railguard.door.features import DoorFeatureTransformer
from railguard.door.segmentation import DoorSegment


class OperationThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Separate mean-current thresholds for Open and Close cycles."""

    def fit(self, X: Sequence[DoorSegment], y: Sequence[int]):
        segments = list(X)
        labels = np.asarray(y, dtype=int)
        self.thresholds_: dict[str, float] = {}
        self.directions_: dict[str, int] = {}
        self.scales_: dict[str, float] = {}
        for operation in ("Open", "Close"):
            indices = [i for i, segment in enumerate(segments) if segment.operation == operation]
            values = np.array(
                [segments[i].frame[CURRENT_COLUMN].mean() for i in indices], dtype=float
            )
            targets = labels[indices]
            unique = np.unique(values)
            candidates = np.concatenate(
                (
                    [np.nextafter(unique[0], -np.inf)],
                    (unique[:-1] + unique[1:]) / 2.0,
                    [np.nextafter(unique[-1], np.inf)],
                )
            )
            best: tuple[int, int, float] | None = None
            for threshold in candidates:
                for direction in (1, -1):
                    predicted = (direction * values > direction * threshold).astype(int)
                    correct = int(np.sum(predicted == targets))
                    recall = int(np.sum((predicted == 1) & (targets == 1)))
                    candidate = (correct, recall, -float(threshold))
                    if best is None or candidate > best:
                        best = candidate
                        self.thresholds_[operation] = float(threshold)
                        self.directions_[operation] = direction
            scale = float(np.std(values))
            self.scales_[operation] = scale if scale > 1e-12 else 1.0
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def decision_function(self, X: Sequence[DoorSegment]) -> np.ndarray:
        scores = []
        for segment in X:
            value = float(segment.frame[CURRENT_COLUMN].mean())
            operation = segment.operation
            scores.append(
                self.directions_[operation]
                * (value - self.thresholds_[operation])
                / self.scales_[operation]
            )
        return np.asarray(scores)

    def predict_proba(self, X: Sequence[DoorSegment]) -> np.ndarray:
        positive = expit(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Sequence[DoorSegment]) -> np.ndarray:
        return (self.decision_function(X) > 0.0).astype(int)


def build_model(name: str, parameter: float | None, template_points: int) -> BaseEstimator:
    """Build one bounded model candidate."""

    if name == "logistic_minimal":
        return Pipeline(
            [
                ("features", DoorFeatureTransformer("minimal", template_points)),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(parameter),
                        solver="lbfgs",
                        max_iter=2_000,
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "logistic_physics":
        return Pipeline(
            [
                ("features", DoorFeatureTransformer("physics", template_points)),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(parameter),
                        solver="lbfgs",
                        max_iter=2_000,
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "linear_svm":
        return Pipeline(
            [
                ("features", DoorFeatureTransformer("physics", template_points)),
                (
                    "classifier",
                    LinearSVC(C=float(parameter), class_weight=None, random_state=42),
                ),
            ]
        )
    if name == "shallow_boosting":
        return Pipeline(
            [
                ("features", DoorFeatureTransformer("physics", template_points)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        loss="log_loss",
                        learning_rate=0.05,
                        max_iter=100,
                        max_depth=2,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown Door model {name!r}")


def model_scores(model: BaseEstimator, X: Sequence[DoorSegment]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)
