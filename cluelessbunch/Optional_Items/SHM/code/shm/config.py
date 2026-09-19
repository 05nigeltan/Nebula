"""Configuration shared by SHM feature extraction and training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShmConfig:
    """Bounded, reproducible configuration for the small labelled dataset."""

    expected_samples: int = 581_120
    exponents: tuple[float, ...] = (4.0, 4.5, 5.0, 5.5, 6.0)
    calibration_modes: tuple[str, ...] = ("fixed", "mape", "fitted")
    inner_folds: int = 5
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0)
    huber_epsilons: tuple[float, ...] = (1.35, 1.5)
    residual_shrinkages: tuple[float, ...] = (0.25, 0.5, 1.0)
    amplitude_bins: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    random_state: int = 42
    positive_floor: float = 1e-12
    hybrid_min_relative_improvement: float = 0.10
    hybrid_min_fold_win_rate: float = 0.70
    hybrid_max_p90_increase: float = 0.01
