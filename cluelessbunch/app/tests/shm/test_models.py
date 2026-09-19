from __future__ import annotations

import numpy as np
import pandas as pd

from railguard.shm.config import ShmConfig
from railguard.shm.models import (
    FittedDamageModel,
    PhysicsSpec,
    RainflowCalibrator,
    ResidualSpec,
    cross_fitted_physics_predictions,
)


def test_fixed_calibrator_recovers_proportional_damage() -> None:
    basis = np.log(np.array([1.0, 2.0, 4.0, 8.0]))
    target = 0.25 * np.exp(basis)
    model = RainflowCalibrator("fixed").fit(basis, target)
    np.testing.assert_allclose(model.predict(basis), target)
    assert model.slope_ == 1.0


def test_mape_calibrator_uses_exact_weighted_median_scale() -> None:
    basis = np.array([1.0, 2.0, 4.0])
    target = np.array([2.0, 10.0, 8.0])
    model = RainflowCalibrator("mape").fit(np.log(basis), target)
    assert model.slope_ == 1.0
    np.testing.assert_allclose(np.exp(model.intercept_), 2.0)


def test_cross_fitted_residual_targets_exclude_each_file(monkeypatch) -> None:
    frame = pd.DataFrame({"log_b_m_5": np.log([1.0, 2.0, 3.0, 4.0])})
    target = np.array([0.1, 0.2, 0.3, 0.4])
    observed_sizes: list[int] = []
    original = RainflowCalibrator.fit

    def recording_fit(self, log_basis, y):
        observed_sizes.append(len(y))
        return original(self, log_basis, y)

    monkeypatch.setattr(RainflowCalibrator, "fit", recording_fit)
    predictions = cross_fitted_physics_predictions(frame, target, PhysicsSpec(5.0, "fixed"))
    assert observed_sizes == [3, 3, 3, 3]
    assert np.all(predictions > 0)


def test_lambda_zero_equivalent_model_is_physics_only() -> None:
    frame = pd.DataFrame(
        {
            "log_b_m_5": np.log([1.0, 2.0, 3.0, 4.0]),
            "rf_bin_0_count_share": [0.2, 0.3, 0.4, 0.5],
        }
    )
    target = np.array([0.1, 0.2, 0.3, 0.4])
    config = ShmConfig()
    pure = FittedDamageModel(PhysicsSpec(5.0, "fixed"), ResidualSpec("none"), config)
    zero = FittedDamageModel(
        PhysicsSpec(5.0, "fixed"),
        ResidualSpec("ridge", 10.0, 0.0, "spectrum"),
        config,
    )
    np.testing.assert_allclose(
        pure.fit(frame, target).predict(frame), zero.fit(frame, target).predict(frame)
    )
