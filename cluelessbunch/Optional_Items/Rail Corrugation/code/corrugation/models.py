"""Side-symmetric linear margin classifier and decision-threshold helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from railguard.corrugation.config import NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL
from railguard.corrugation.metric import CorrugationScore, score_corrugation


@dataclass(frozen=True)
class ModelSpec:
    feature_family: str
    c_value: float
    positive_weight_multiplier: float
    include_raw_speed: bool
    threshold: float = 0.0
    side_i_bias: float = 0.0


def available_side_bases(frame: pd.DataFrame) -> list[str]:
    first = {column.removeprefix("side_i_") for column in frame if column.startswith("side_i_")}
    second = {column.removeprefix("side_ii_") for column in frame if column.startswith("side_ii_")}
    if first != second:
        raise ValueError("Side I and Side II feature schemas differ")
    return sorted(first)


def feature_bases(frame: pd.DataFrame, family: str) -> list[str]:
    bases = available_side_bases(frame)
    if family == "all":
        return bases
    time_tokens = ("_rms_", "_peak_", "_crest_", "_kurtosis_")
    if family == "time":
        return [
            base
            for base in bases
            if "_spatial_" not in base
            and any(token in f"_{base}" for token in time_tokens)
        ]
    if family == "compact":
        selected = []
        for base in bases:
            is_time = "_spatial_" not in base and any(
                token in f"_{base}" for token in time_tokens
            )
            is_wave = "wave_band" in base and "_max" not in base
            is_shape = any(
                token in base
                for token in ("spectral_entropy_median", "spectral_concentration_median")
            )
            is_consensus = "wave_consensus" in base
            if is_time or is_wave or is_shape or is_consensus:
                selected.append(base)
        return selected
    raise ValueError(f"Unknown feature family {family!r}")


def build_paired_matrix(
    frame: pd.DataFrame,
    bases: list[str],
    include_raw_speed: bool,
) -> np.ndarray:
    first = frame[[f"side_i_{base}" for base in bases]].to_numpy(float)
    second = frame[[f"side_ii_{base}" for base in bases]].to_numpy(float)
    global_names = ["wavelength_valid"]
    if include_raw_speed:
        global_names.extend(["tach_transitions", "speed_mps", "tach_duty"])
    global_values = frame[global_names].to_numpy(float)

    def local(own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return np.column_stack((own, other, own - other, np.abs(own - other), global_values))

    matrix = np.empty((len(frame) * 2, len(bases) * 4 + len(global_names)), dtype=float)
    matrix[0::2] = local(first, second)
    matrix[1::2] = local(second, first)
    return matrix


def paired_targets(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    target = np.zeros(len(labels) * 2, dtype=int)
    target[0::2] = labels == SIDE_I_LABEL
    target[1::2] = labels == SIDE_II_LABEL
    return target


def labels_from_scores(
    side_i_score: np.ndarray,
    side_ii_score: np.ndarray,
    threshold: float,
    side_i_bias: float = 0.0,
) -> np.ndarray:
    first = np.asarray(side_i_score, dtype=float) + side_i_bias
    second = np.asarray(side_ii_score, dtype=float)
    result = np.full(len(first), NORMAL_LABEL, dtype=object)
    fault = np.maximum(first, second) >= threshold
    result[fault & (first >= second)] = SIDE_I_LABEL
    result[fault & (second > first)] = SIDE_II_LABEL
    return result


def choose_threshold(
    truth: np.ndarray,
    side_i_score: np.ndarray,
    side_ii_score: np.ndarray,
    grid_size: int,
    minimum_side_i_recall: float = 0.0,
    minimum_side_ii_recall: float = 0.0,
    side_i_bias_values: tuple[float, ...] = (0.0,),
) -> tuple[float, float, CorrugationScore]:
    best_key: tuple[float, float, float, float] | None = None
    best_threshold = 0.0
    best_bias = 0.0
    best_score: CorrugationScore | None = None
    for bias in side_i_bias_values:
        maximum = np.maximum(side_i_score + bias, side_ii_score)
        quantiles = np.linspace(0.0, 1.0, min(grid_size, len(maximum) + 1))
        candidates = np.unique(
            np.concatenate(
                (
                    np.quantile(maximum, quantiles),
                    np.asarray([0.0, np.nextafter(np.min(maximum), -np.inf)]),
                )
            )
        )
        for threshold in candidates:
            predicted = labels_from_scores(
                side_i_score, side_ii_score, float(threshold), float(bias)
            )
            score = score_corrugation(truth, predicted)
            fault_recall = min(
                score.per_class_recall[SIDE_I_LABEL], score.per_class_recall[SIDE_II_LABEL]
            )
            recall_gate = float(
                score.per_class_recall[SIDE_I_LABEL] >= minimum_side_i_recall
                and score.per_class_recall[SIDE_II_LABEL] >= minimum_side_ii_recall
            )
            key = (recall_gate, score.macro_f1, fault_recall, -abs(float(bias)))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
                best_bias = float(bias)
                best_score = score
    assert best_score is not None
    return best_threshold, best_bias, best_score


class FittedCorrugationModel:
    """Serializable shared side detector with a three-class decision rule."""

    def __init__(self, spec: ModelSpec, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = random_state

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> FittedCorrugationModel:
        self.feature_bases_ = feature_bases(frame, self.spec.feature_family)
        matrix = build_paired_matrix(frame, self.feature_bases_, self.spec.include_raw_speed)
        target = paired_targets(labels)
        negative_count = int(np.sum(target == 0))
        positive_count = int(np.sum(target == 1))
        total = len(target)
        class_weight = {
            0: total / (2.0 * negative_count),
            1: total / (2.0 * positive_count) * self.spec.positive_weight_multiplier,
        }
        self.scaler_ = StandardScaler().fit(matrix)
        transformed = self.scaler_.transform(matrix)
        self.classifier_ = LinearSVC(
            C=self.spec.c_value,
            class_weight=class_weight,
            dual="auto",
            max_iter=20_000,
            random_state=self.random_state,
        ).fit(transformed, target)
        return self

    def decision_scores(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = build_paired_matrix(frame, self.feature_bases_, self.spec.include_raw_speed)
        scores = self.classifier_.decision_function(self.scaler_.transform(matrix))
        return scores[0::2], scores[1::2]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        first, second = self.decision_scores(frame)
        return labels_from_scores(first, second, self.spec.threshold, self.spec.side_i_bias)
