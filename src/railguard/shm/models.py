"""Physics calibration and optional regularized residual correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from railguard.shm.config import ShmConfig
from railguard.shm.rainflow_features import feature_columns_for_group


@dataclass(frozen=True)
class PhysicsSpec:
    exponent: float
    calibration_mode: str


@dataclass(frozen=True)
class ResidualSpec:
    estimator: str
    parameter: float = 0.0
    shrinkage: float = 0.0
    feature_group: str = "none"

    @property
    def name(self) -> str:
        if self.estimator == "none":
            return "none"
        return f"{self.estimator}_{self.parameter:g}_{self.shrinkage:g}_{self.feature_group}"


class RainflowCalibrator:
    """Calibrate a rainflow power sum to positive damage in log space."""

    def __init__(self, mode: str = "fitted") -> None:
        if mode not in {"fixed", "mape", "fitted"}:
            raise ValueError(f"Unknown calibration mode {mode!r}")
        self.mode = mode

    def fit(self, log_basis: np.ndarray, target: np.ndarray) -> RainflowCalibrator:
        x = np.asarray(log_basis, dtype=float)
        log_y = np.log(np.asarray(target, dtype=float))
        if self.mode == "fixed":
            self.slope_ = 1.0
            self.intercept_ = float(np.mean(log_y - x))
        elif self.mode == "mape":
            basis = np.exp(x)
            scale_candidates = np.asarray(target, dtype=float) / basis
            weights = basis / np.asarray(target, dtype=float)
            order = np.argsort(scale_candidates)
            cumulative = np.cumsum(weights[order])
            median_index = int(np.searchsorted(cumulative, 0.5 * np.sum(weights), side="left"))
            scale = float(scale_candidates[order][median_index])
            self.slope_ = 1.0
            self.intercept_ = float(np.log(scale))
        else:
            design = np.column_stack((np.ones(len(x)), x))
            coefficients, *_ = np.linalg.lstsq(design, log_y, rcond=None)
            self.intercept_ = float(coefficients[0])
            self.slope_ = float(coefficients[1])
        return self

    def predict(self, log_basis: np.ndarray) -> np.ndarray:
        return np.exp(self.intercept_ + self.slope_ * np.asarray(log_basis, dtype=float))


def log_basis_column(exponent: float) -> str:
    return f"log_b_m_{exponent:g}"


def cross_fitted_physics_predictions(
    frame: pd.DataFrame,
    target: np.ndarray,
    spec: PhysicsSpec,
) -> np.ndarray:
    """Leave each training file out of its own residual target."""

    y = np.asarray(target, dtype=float)
    basis = frame[log_basis_column(spec.exponent)].to_numpy(float)
    predictions = np.empty(len(y), dtype=float)
    indices = np.arange(len(y))
    for held_out in indices:
        training = indices != held_out
        calibrator = RainflowCalibrator(spec.calibration_mode).fit(basis[training], y[training])
        predictions[held_out] = calibrator.predict(basis[[held_out]])[0]
    return predictions


def build_residual_regressor(spec: ResidualSpec):
    if spec.estimator == "ridge":
        return Ridge(alpha=spec.parameter)
    if spec.estimator == "huber":
        return HuberRegressor(epsilon=spec.parameter, alpha=0.01, max_iter=1000)
    raise ValueError(f"Unknown residual estimator {spec.estimator!r}")


class FittedDamageModel:
    """Serializable positive damage model with an inspectable physics backbone."""

    def __init__(
        self,
        physics_spec: PhysicsSpec,
        residual_spec: ResidualSpec,
        config: ShmConfig,
    ) -> None:
        self.physics_spec = physics_spec
        self.residual_spec = residual_spec
        self.config = config

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> FittedDamageModel:
        y = np.asarray(target, dtype=float)
        basis = frame[log_basis_column(self.physics_spec.exponent)].to_numpy(float)
        self.physics_ = RainflowCalibrator(self.physics_spec.calibration_mode).fit(basis, y)
        self.feature_names_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.residual_model_ = None
        if self.residual_spec.estimator != "none" and self.residual_spec.shrinkage > 0:
            self.feature_names_ = feature_columns_for_group(
                frame.columns, self.residual_spec.feature_group
            )
            if not self.feature_names_:
                raise ValueError("Residual feature selection produced no columns")
            cross_fitted = cross_fitted_physics_predictions(frame, y, self.physics_spec)
            residual_target = np.log(y / cross_fitted)
            self.scaler_ = StandardScaler().fit(frame[self.feature_names_].to_numpy(float))
            transformed = self.scaler_.transform(frame[self.feature_names_].to_numpy(float))
            self.residual_model_ = build_residual_regressor(self.residual_spec)
            self.residual_model_.fit(transformed, residual_target)
        return self

    def predict_components(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        basis = frame[log_basis_column(self.physics_spec.exponent)].to_numpy(float)
        physics = self.physics_.predict(basis)
        correction = np.ones(len(frame), dtype=float)
        if self.residual_model_ is not None and self.scaler_ is not None:
            transformed = self.scaler_.transform(frame[self.feature_names_].to_numpy(float))
            residual = self.residual_model_.predict(transformed)
            correction = np.exp(self.residual_spec.shrinkage * residual)
        return physics, correction

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        physics, correction = self.predict_components(frame)
        return np.maximum(physics * correction, self.config.positive_floor)
