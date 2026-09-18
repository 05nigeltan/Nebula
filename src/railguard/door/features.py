"""Leakage-safe cycle-level feature extraction for Door models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from railguard.door.config import (
    BACK_EMF_COLUMN,
    CURRENT_COLUMN,
    POSITION_COLUMN,
    VOLTAGE_COLUMN,
)
from railguard.door.segmentation import DoorSegment

MINIMAL_FEATURES = (
    "duration_s",
    "current_mean",
    "current_p95",
    "mid_current_mean",
    "mid_current_max",
    "template_residual_mean",
    "template_residual_area",
)

PHYSICS_FEATURES = (
    "duration_s",
    "current_mean",
    "current_std",
    "current_median",
    "current_p90",
    "current_p95",
    "current_max",
    "current_integral",
    "voltage_mean",
    "voltage_std",
    "voltage_p95",
    "back_emf_mean",
    "back_emf_std",
    "back_emf_p95",
    "current_voltage_ratio",
    "net_effort_ratio",
    "position_travel",
    "mean_abs_velocity",
    "max_abs_velocity",
    "low_speed_current",
    "mid_current_mean",
    "mid_current_max",
    "template_residual_mean",
    "template_residual_p95",
    "template_residual_area",
)


def _progress_and_profile(segment: DoorSegment, values: np.ndarray, points: int) -> np.ndarray:
    position = segment.frame[POSITION_COLUMN].to_numpy(dtype=float)
    delta = position[-1] - position[0]
    if abs(delta) < 1e-9:
        progress = np.linspace(0.0, 1.0, len(position))
    else:
        progress = np.clip((position - position[0]) / delta, 0.0, 1.0)
        progress = np.maximum.accumulate(progress)
        if progress[-1] <= 0:
            progress = np.linspace(0.0, 1.0, len(position))
        else:
            progress = progress / progress[-1]

    unique_progress, unique_indices = np.unique(progress, return_index=True)
    unique_values = values[unique_indices]
    if len(unique_progress) < 2:
        return np.full(points, float(np.mean(values)))
    grid = np.linspace(0.0, 1.0, points)
    return np.interp(grid, unique_progress, unique_values)


def current_progress_profile(segment: DoorSegment, points: int = 64) -> np.ndarray:
    current = segment.frame[CURRENT_COLUMN].to_numpy(dtype=float)
    return _progress_and_profile(segment, current, points)


def extract_cycle_features(
    segment: DoorSegment,
    *,
    normal_template: np.ndarray | None,
    template_points: int,
) -> dict[str, float]:
    """Extract fixed, physically motivated features from one complete cycle."""

    frame = segment.frame
    current = frame[CURRENT_COLUMN].to_numpy(dtype=float)
    voltage = frame[VOLTAGE_COLUMN].to_numpy(dtype=float)
    back_emf = frame[BACK_EMF_COLUMN].to_numpy(dtype=float)
    position = frame[POSITION_COLUMN].to_numpy(dtype=float)
    time_s = (
        frame["_parsed_time"].astype("int64").to_numpy(dtype=float)
        - float(frame["_parsed_time"].astype("int64").iloc[0])
    ) / 1e9
    duration_s = max(float(time_s[-1]), 1e-9)

    dt = np.diff(time_s)
    dp = np.diff(position)
    valid_dt = np.where(dt > 0, dt, np.nan)
    velocity = np.abs(dp / valid_dt)
    finite_velocity = velocity[np.isfinite(velocity)]
    if finite_velocity.size == 0:
        finite_velocity = np.array([0.0])

    profile = current_progress_profile(segment, template_points)
    progress_grid = np.linspace(0.0, 1.0, template_points)
    mid_mask = (progress_grid >= 0.35) & (progress_grid <= 0.55)
    mid_values = profile[mid_mask]

    low_speed_cutoff = float(np.quantile(finite_velocity, 0.25))
    sample_velocity = np.concatenate(([finite_velocity[0]], finite_velocity))
    if len(sample_velocity) != len(current):
        sample_velocity = np.resize(sample_velocity, len(current))
    low_speed_values = current[sample_velocity <= low_speed_cutoff]
    if low_speed_values.size == 0:
        low_speed_values = current

    if normal_template is None:
        residual = np.zeros_like(profile)
    else:
        residual = np.maximum(profile - normal_template, 0.0)

    voltage_mean = float(np.mean(np.abs(voltage)))
    net_voltage = np.maximum(np.abs(voltage) - np.abs(back_emf), 1.0)
    return {
        "duration_s": duration_s,
        "current_mean": float(np.mean(current)),
        "current_std": float(np.std(current)),
        "current_median": float(np.median(current)),
        "current_p90": float(np.quantile(current, 0.90)),
        "current_p95": float(np.quantile(current, 0.95)),
        "current_max": float(np.max(current)),
        "current_integral": float(np.trapezoid(current, time_s)),
        "voltage_mean": float(np.mean(voltage)),
        "voltage_std": float(np.std(voltage)),
        "voltage_p95": float(np.quantile(voltage, 0.95)),
        "back_emf_mean": float(np.mean(back_emf)),
        "back_emf_std": float(np.std(back_emf)),
        "back_emf_p95": float(np.quantile(back_emf, 0.95)),
        "current_voltage_ratio": float(np.mean(current)) / max(voltage_mean, 1.0),
        "net_effort_ratio": float(np.mean(current / net_voltage)),
        "position_travel": float(abs(position[-1] - position[0])),
        "mean_abs_velocity": float(np.mean(finite_velocity)),
        "max_abs_velocity": float(np.max(finite_velocity)),
        "low_speed_current": float(np.mean(low_speed_values)),
        "mid_current_mean": float(np.mean(mid_values)),
        "mid_current_max": float(np.max(mid_values)),
        "template_residual_mean": float(np.mean(residual)),
        "template_residual_p95": float(np.quantile(residual, 0.95)),
        "template_residual_area": float(np.trapezoid(residual, progress_grid)),
    }


class DoorFeatureTransformer(BaseEstimator, TransformerMixin):
    """Fit Normal templates and operation-specific scaling inside each fold."""

    def __init__(self, feature_set: str = "physics", template_points: int = 64):
        self.feature_set = feature_set
        self.template_points = template_points

    def _selected_names(self) -> tuple[str, ...]:
        if self.feature_set == "minimal":
            return MINIMAL_FEATURES
        if self.feature_set == "physics":
            return PHYSICS_FEATURES
        raise ValueError(f"Unknown feature_set {self.feature_set!r}")

    def fit(self, X: Sequence[DoorSegment], y: Sequence[int]):
        segments = list(X)
        labels = np.asarray(y, dtype=int)
        if len(segments) != len(labels):
            raise ValueError("X and y must contain the same number of cycles")

        self.templates_: dict[str, np.ndarray] = {}
        for operation in ("Open", "Close"):
            profiles = [
                current_progress_profile(segment, self.template_points)
                for segment, label in zip(segments, labels, strict=True)
                if segment.operation == operation and label == 0
            ]
            if not profiles:
                raise ValueError(f"No Normal {operation} cycles are available to fit a template")
            self.templates_[operation] = np.median(np.vstack(profiles), axis=0)

        names = self._selected_names()
        raw = self._raw_matrix(segments, names)
        operations = np.array([segment.operation for segment in segments])
        self.operation_means_: dict[str, np.ndarray] = {}
        self.operation_scales_: dict[str, np.ndarray] = {}
        for operation in ("Open", "Close"):
            values = raw[operations == operation]
            if len(values) == 0:
                raise ValueError(f"No {operation} cycles are available for scaling")
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale[scale < 1e-12] = 1.0
            self.operation_means_[operation] = mean
            self.operation_scales_[operation] = scale
        self.feature_names_inferred_ = (*names, "operation_open")
        return self

    def _raw_matrix(self, segments: Sequence[DoorSegment], names: tuple[str, ...]) -> np.ndarray:
        rows = []
        for segment in segments:
            template = getattr(self, "templates_", {}).get(segment.operation)
            features = extract_cycle_features(
                segment,
                normal_template=template,
                template_points=self.template_points,
            )
            rows.append([features[name] for name in names])
        return np.asarray(rows, dtype=float)

    def transform(self, X: Sequence[DoorSegment]) -> np.ndarray:
        segments = list(X)
        names = self._selected_names()
        raw = self._raw_matrix(segments, names)
        scaled = np.empty_like(raw)
        operation_flag = np.empty((len(segments), 1), dtype=float)
        for index, segment in enumerate(segments):
            scaled[index] = (
                raw[index] - self.operation_means_[segment.operation]
            ) / self.operation_scales_[segment.operation]
            operation_flag[index, 0] = 1.0 if segment.operation == "Open" else 0.0
        result = np.hstack((scaled, operation_flag))
        if not np.isfinite(result).all():
            raise ValueError("Door feature extraction produced a non-finite value")
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        return np.asarray(self.feature_names_inferred_, dtype=object)
