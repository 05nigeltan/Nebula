"""Nested file-level validation and the hybrid deployment gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from railguard.shm.config import ShmConfig
from railguard.shm.metric import ShmMetrics, score_shm
from railguard.shm.models import (
    FittedDamageModel,
    PhysicsSpec,
    RainflowCalibrator,
    ResidualSpec,
    build_residual_regressor,
    cross_fitted_physics_predictions,
    log_basis_column,
)
from railguard.shm.rainflow_features import feature_columns_for_group


@dataclass(frozen=True)
class ValidationResult:
    median_predictions: np.ndarray
    physics_predictions: np.ndarray
    hybrid_predictions: np.ndarray
    selected_predictions: np.ndarray
    median_metrics: ShmMetrics
    physics_metrics: ShmMetrics
    hybrid_metrics: ShmMetrics
    selected_metrics: ShmMetrics
    hybrid_accepted: bool
    gate: dict[str, float | bool]
    fold_results: list[dict[str, Any]]
    final_physics_spec: PhysicsSpec
    final_residual_spec: ResidualSpec


def residual_candidates(config: ShmConfig) -> list[ResidualSpec]:
    """Return stable first-release candidates.

    Huber remains implemented for later experiments, but is deliberately excluded here after
    repeated optimizer non-convergence on the small, collinear feature matrix.
    """

    candidates = [ResidualSpec("none")]
    groups = ("spectrum", "spectrum_chronology", "compact_all")
    for group in groups:
        for shrinkage in config.residual_shrinkages:
            candidates.extend(
                ResidualSpec("ridge", alpha, shrinkage, group)
                for alpha in config.ridge_alphas
            )
    return candidates


def _physics_loo_predictions(
    frame: pd.DataFrame,
    target: np.ndarray,
    spec: PhysicsSpec,
) -> np.ndarray:
    return cross_fitted_physics_predictions(frame.reset_index(drop=True), target, spec)


def select_physics_spec(
    frame: pd.DataFrame,
    target: np.ndarray,
    config: ShmConfig,
) -> tuple[PhysicsSpec, float]:
    best: tuple[float, int, float, PhysicsSpec] | None = None
    for exponent in config.exponents:
        for mode in config.calibration_modes:
            spec = PhysicsSpec(exponent, mode)
            predictions = _physics_loo_predictions(frame, target, spec)
            mape = score_shm(target, predictions).mape
            complexity = {"fixed": 0, "mape": 1, "fitted": 2}[mode]
            candidate = (mape, complexity, abs(exponent - 5.0), spec)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    return best[3], best[0]


def _predict_inner_candidates(
    train_frame: pd.DataFrame,
    train_target: np.ndarray,
    validation_frame: pd.DataFrame,
    physics_spec: PhysicsSpec,
    candidates: list[ResidualSpec],
) -> dict[str, np.ndarray]:
    basis_column = log_basis_column(physics_spec.exponent)
    physics = RainflowCalibrator(physics_spec.calibration_mode).fit(
        train_frame[basis_column].to_numpy(float), train_target
    )
    validation_physics = physics.predict(validation_frame[basis_column].to_numpy(float))
    cross_fitted = cross_fitted_physics_predictions(
        train_frame.reset_index(drop=True), train_target, physics_spec
    )
    residual_target = np.log(train_target / cross_fitted)
    result: dict[str, np.ndarray] = {}
    for candidate in candidates:
        if candidate.estimator == "none":
            result[candidate.name] = validation_physics.copy()
            continue
        feature_names = feature_columns_for_group(train_frame.columns, candidate.feature_group)
        scaler = StandardScaler().fit(train_frame[feature_names].to_numpy(float))
        regressor = build_residual_regressor(candidate)
        regressor.fit(scaler.transform(train_frame[feature_names].to_numpy(float)), residual_target)
        residual = regressor.predict(
            scaler.transform(validation_frame[feature_names].to_numpy(float))
        )
        result[candidate.name] = validation_physics * np.exp(candidate.shrinkage * residual)
    return result


def select_residual_spec(
    frame: pd.DataFrame,
    target: np.ndarray,
    physics_spec: PhysicsSpec,
    config: ShmConfig,
) -> tuple[ResidualSpec, float]:
    candidates = residual_candidates(config)
    predictions = {candidate.name: np.empty(len(target), dtype=float) for candidate in candidates}
    folds = min(config.inner_folds, len(target))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    for train_index, validation_index in splitter.split(frame):
        fold_predictions = _predict_inner_candidates(
            frame.iloc[train_index].reset_index(drop=True),
            target[train_index],
            frame.iloc[validation_index].reset_index(drop=True),
            physics_spec,
            candidates,
        )
        for name, values in fold_predictions.items():
            predictions[name][validation_index] = values
    complexity = {"none": 0, "ridge": 1, "huber": 2}
    best: tuple[float, int, float, ResidualSpec] | None = None
    for candidate in candidates:
        mape = score_shm(target, predictions[candidate.name]).mape
        key = (mape, complexity[candidate.estimator], candidate.shrinkage, candidate)
        if best is None or key[:3] < best[:3]:
            best = key
    assert best is not None
    return best[3], best[0]


def nested_validate(frame: pd.DataFrame, target: np.ndarray, config: ShmConfig) -> ValidationResult:
    """Outer leave-one-file-out evaluation; no held-out file participates in tuning."""

    y = np.asarray(target, dtype=float)
    sample_count = len(y)
    median_predictions = np.empty(sample_count, dtype=float)
    physics_predictions = np.empty(sample_count, dtype=float)
    hybrid_predictions = np.empty(sample_count, dtype=float)
    fold_results: list[dict[str, Any]] = []
    all_indices = np.arange(sample_count)

    for held_out in all_indices:
        training = all_indices[all_indices != held_out]
        train_frame = frame.iloc[training].reset_index(drop=True)
        validation_frame = frame.iloc[[held_out]].reset_index(drop=True)
        train_target = y[training]
        physics_spec, inner_physics_mape = select_physics_spec(train_frame, train_target, config)
        residual_spec, inner_residual_mape = select_residual_spec(
            train_frame, train_target, physics_spec, config
        )
        physics_model = FittedDamageModel(physics_spec, ResidualSpec("none"), config).fit(
            train_frame, train_target
        )
        hybrid_model = FittedDamageModel(physics_spec, residual_spec, config).fit(
            train_frame, train_target
        )
        median_predictions[held_out] = float(np.median(train_target))
        physics_predictions[held_out] = physics_model.predict(validation_frame)[0]
        hybrid_predictions[held_out] = hybrid_model.predict(validation_frame)[0]
        fold_results.append(
            {
                "held_out_index": int(held_out),
                "held_out_file": str(frame.iloc[held_out]["file_id"]),
                "physics_exponent": physics_spec.exponent,
                "calibration_mode": physics_spec.calibration_mode,
                "residual_model": residual_spec.name,
                "inner_physics_mape": inner_physics_mape,
                "inner_selected_mape": inner_residual_mape,
                "truth": y[held_out],
                "physics_prediction": physics_predictions[held_out],
                "hybrid_prediction": hybrid_predictions[held_out],
            }
        )

    median_metrics = score_shm(y, median_predictions)
    physics_metrics = score_shm(y, physics_predictions)
    hybrid_metrics = score_shm(y, hybrid_predictions)
    physics_ape = np.abs(physics_predictions - y) / y
    hybrid_ape = np.abs(hybrid_predictions - y) / y
    relative_improvement = (physics_metrics.mape - hybrid_metrics.mape) / physics_metrics.mape
    fold_win_rate = float(np.mean(hybrid_ape < physics_ape))
    p90_increase = hybrid_metrics.p90_ape - physics_metrics.p90_ape
    hybrid_accepted = bool(
        relative_improvement >= config.hybrid_min_relative_improvement
        and fold_win_rate >= config.hybrid_min_fold_win_rate
        and p90_increase <= config.hybrid_max_p90_increase
    )
    gate: dict[str, float | bool] = {
        "relative_mape_improvement": float(relative_improvement),
        "required_relative_improvement": config.hybrid_min_relative_improvement,
        "fold_win_rate": fold_win_rate,
        "required_fold_win_rate": config.hybrid_min_fold_win_rate,
        "p90_ape_increase": float(p90_increase),
        "maximum_p90_increase": config.hybrid_max_p90_increase,
        "accepted": hybrid_accepted,
    }
    final_physics_spec, _ = select_physics_spec(frame.reset_index(drop=True), y, config)
    final_residual_spec, _ = select_residual_spec(
        frame.reset_index(drop=True), y, final_physics_spec, config
    )
    if not hybrid_accepted:
        final_residual_spec = ResidualSpec("none")
    selected_predictions = hybrid_predictions if hybrid_accepted else physics_predictions
    selected_metrics = hybrid_metrics if hybrid_accepted else physics_metrics
    return ValidationResult(
        median_predictions=median_predictions,
        physics_predictions=physics_predictions,
        hybrid_predictions=hybrid_predictions,
        selected_predictions=selected_predictions,
        median_metrics=median_metrics,
        physics_metrics=physics_metrics,
        hybrid_metrics=hybrid_metrics,
        selected_metrics=selected_metrics,
        hybrid_accepted=hybrid_accepted,
        gate=gate,
        fold_results=fold_results,
        final_physics_spec=final_physics_spec,
        final_residual_spec=final_residual_spec,
    )
