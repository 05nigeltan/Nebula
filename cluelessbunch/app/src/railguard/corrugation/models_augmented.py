"""Weighted side-symmetric classifier for grouped sensor-view augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from railguard.corrugation.models import (
    ModelSpec,
    build_paired_matrix,
    feature_bases,
    labels_from_scores,
    paired_targets,
)


class FittedAugmentedCorrugationModel:
    """A v1-compatible estimator that conserves weight across correlated views."""

    def __init__(self, spec: ModelSpec, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = random_state

    def fit(
        self,
        frame: pd.DataFrame,
        labels: np.ndarray,
        view_weights: np.ndarray | None = None,
    ) -> FittedAugmentedCorrugationModel:
        labels = np.asarray(labels)
        if len(frame) != len(labels):
            raise ValueError("Feature and label counts differ")
        weights = (
            np.ones(len(frame), dtype=float)
            if view_weights is None
            else np.asarray(view_weights, dtype=float)
        )
        if weights.shape != (len(frame),) or not np.all(np.isfinite(weights)):
            raise ValueError("view_weights must be one finite value per feature row")
        if np.any(weights <= 0):
            raise ValueError("view_weights must be strictly positive")

        self.feature_bases_ = feature_bases(frame, self.spec.feature_family)
        matrix = build_paired_matrix(frame, self.feature_bases_, self.spec.include_raw_speed)
        target = paired_targets(labels)
        paired_weights = np.repeat(weights, 2)
        negative_mass = float(np.sum(paired_weights[target == 0]))
        positive_mass = float(np.sum(paired_weights[target == 1]))
        if negative_mass <= 0 or positive_mass <= 0:
            raise ValueError("Training requires both positive and negative paired targets")
        total_mass = negative_mass + positive_mass
        class_weight = {
            0: total_mass / (2.0 * negative_mass),
            1: total_mass / (2.0 * positive_mass) * self.spec.positive_weight_multiplier,
        }

        self.scaler_ = StandardScaler().fit(matrix, sample_weight=paired_weights)
        transformed = self.scaler_.transform(matrix)
        self.classifier_ = LinearSVC(
            C=self.spec.c_value,
            class_weight=class_weight,
            dual="auto",
            max_iter=20_000,
            random_state=self.random_state,
        ).fit(transformed, target, sample_weight=paired_weights)
        return self

    def decision_scores(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = build_paired_matrix(frame, self.feature_bases_, self.spec.include_raw_speed)
        scores = self.classifier_.decision_function(self.scaler_.transform(matrix))
        return scores[0::2], scores[1::2]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        first, second = self.decision_scores(frame)
        return labels_from_scores(first, second, self.spec.threshold, self.spec.side_i_bias)
