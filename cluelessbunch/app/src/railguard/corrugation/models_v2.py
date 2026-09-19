"""Compact linear/RBF side detectors for corrugation v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from railguard.corrugation.config import NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL
from railguard.corrugation.models import available_side_bases, labels_from_scores


@dataclass(frozen=True)
class V2ModelSpec:
    feature_family: str
    kernel: str
    c_value: float
    gamma: str | float = "scale"
    positive_weight_multiplier: float = 1.0
    speed_weight_cap: float = 1.0
    threshold: float = 0.0
    side_i_bias: float = 0.0


def v2_feature_bases(frame: pd.DataFrame, family: str) -> list[str]:
    """Return a bounded, side-symmetric feature block."""

    bases = available_side_bases(frame)
    vibration_spatial = [base for base in bases if base.startswith("vibration_spatial_")]
    all_spatial = [base for base in bases if "_spatial_" in base]
    time_context = [
        base
        for base in bases
        if base
        in {
            "vibration_rms_median",
            "vibration_rms_q90",
            "vibration_crest_q90",
            "vibration_kurtosis_q90",
            "shock_rms_q90",
        }
    ]
    time_features = [
        base
        for base in bases
        if "_spatial_" not in base
        and any(token in f"_{base}" for token in ("_rms_", "_peak_", "_crest_", "_kurtosis_"))
    ]
    def is_spatial_core(base: str) -> bool:
        vibration_core = base.startswith("vibration_spatial_") and (
            ("_band_" in base and base.endswith("_q90"))
            or "_consensus_" in base
            or base.endswith(
                ("peak_prominence_q90", "concentration_q90", "persistence_median")
            )
            or "peak_wavelength_" in base
        )
        shock_auxiliary = base in {
            "shock_spatial_band_0_q90",
            "shock_spatial_peak_prominence_q90",
        }
        return vibration_core or shock_auxiliary

    spatial_core = [base for base in bases if is_spatial_core(base)]
    if family == "time":
        selected = time_features
    elif family == "spatial_vibration":
        selected = vibration_spatial
    elif family == "spatial_core":
        selected = spatial_core + time_context
    elif family == "spatial_time":
        selected = vibration_spatial + time_context
    elif family == "spatial_all":
        selected = all_spatial + time_context
    else:
        raise ValueError(f"Unknown corrugation v2 feature family {family!r}")
    if not selected:
        raise ValueError(f"No features are available for corrugation v2 family {family!r}")
    return sorted(set(selected))


def build_v2_paired_matrix(
    frame: pd.DataFrame,
    bases: list[str],
    global_feature: str = "spatial_valid",
) -> np.ndarray:
    first = frame[[f"side_i_{base}" for base in bases]].to_numpy(float)
    second = frame[[f"side_ii_{base}" for base in bases]].to_numpy(float)
    valid = frame[[global_feature]].to_numpy(float)

    def local(own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return np.column_stack((own, other, own - other, np.abs(own - other), valid))

    matrix = np.empty((len(frame) * 2, len(bases) * 4 + 1), dtype=float)
    matrix[0::2] = local(first, second)
    matrix[1::2] = local(second, first)
    return matrix


def _paired_targets(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    target = np.zeros(len(labels) * 2, dtype=int)
    target[0::2] = labels == SIDE_I_LABEL
    target[1::2] = labels == SIDE_II_LABEL
    return target


def speed_bin_file_weights(frame: pd.DataFrame, cap: float) -> np.ndarray:
    """Equalise speed-bin influence while bounding each file's leverage."""

    if cap <= 1.0:
        return np.ones(len(frame), dtype=float)
    speed_bins = np.digitize(frame["speed_mps"].to_numpy(float), [9.5, 12.0, 15.0])
    unique, counts = np.unique(speed_bins, return_counts=True)
    by_bin = {
        int(bin_id): len(frame) / (len(unique) * int(count))
        for bin_id, count in zip(unique, counts, strict=True)
    }
    weights = np.asarray([by_bin[int(bin_id)] for bin_id in speed_bins], dtype=float)
    weights /= np.mean(weights)
    weights = np.clip(weights, 1.0 / cap, cap)
    return weights / np.mean(weights)


class FittedCorrugationV2:
    """Serializable shared side detector using compact spatial evidence."""

    def __init__(self, spec: V2ModelSpec, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = random_state

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> FittedCorrugationV2:
        self.feature_bases_ = v2_feature_bases(frame, self.spec.feature_family)
        self.global_feature_ = (
            "wavelength_valid" if self.spec.feature_family == "time" else "spatial_valid"
        )
        matrix = build_v2_paired_matrix(frame, self.feature_bases_, self.global_feature_)
        target = _paired_targets(labels)
        negative_count = int(np.sum(target == 0))
        positive_count = int(np.sum(target == 1))
        total = len(target)
        class_weight = {
            0: total / (2.0 * negative_count),
            1: total / (2.0 * positive_count) * self.spec.positive_weight_multiplier,
        }
        file_weights = speed_bin_file_weights(frame, self.spec.speed_weight_cap)
        sample_weight = np.repeat(file_weights, 2)
        self.scaler_ = StandardScaler().fit(matrix)
        transformed = self.scaler_.transform(matrix)
        if self.spec.kernel == "linear":
            self.classifier_ = LinearSVC(
                C=self.spec.c_value,
                class_weight=class_weight,
                dual="auto",
                max_iter=20_000,
                random_state=self.random_state,
            )
        elif self.spec.kernel == "rbf":
            self.classifier_ = SVC(
                C=self.spec.c_value,
                gamma=self.spec.gamma,
                kernel="rbf",
                class_weight=class_weight,
                cache_size=512,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown corrugation v2 kernel {self.spec.kernel!r}")
        self.classifier_.fit(transformed, target, sample_weight=sample_weight)
        return self

    def decision_scores(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = build_v2_paired_matrix(frame, self.feature_bases_, self.global_feature_)
        scores = self.classifier_.decision_function(self.scaler_.transform(matrix))
        return np.asarray(scores[0::2]), np.asarray(scores[1::2])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        first, second = self.decision_scores(frame)
        return labels_from_scores(first, second, self.spec.threshold, self.spec.side_i_bias)

    @property
    def classes_(self) -> tuple[str, str, str]:
        return NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL
