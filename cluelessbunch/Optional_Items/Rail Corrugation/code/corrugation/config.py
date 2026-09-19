"""Configuration and output constants for rail-corrugation modelling."""

from __future__ import annotations

from dataclasses import dataclass

NORMAL_LABEL = "Normal"
SIDE_I_LABEL = "Side I"
SIDE_II_LABEL = "Side II"
VALID_LABELS = (NORMAL_LABEL, SIDE_I_LABEL, SIDE_II_LABEL)
OUTPUT_COLUMNS = ("file_id", "prediction")


@dataclass(frozen=True)
class CorrugationConfig:
    """Bounded configuration shared by extraction, validation, and inference."""

    expected_samples: int = 10_000
    sample_rate_hz: float = 10_000.0
    tach_teeth: int = 90
    wheel_diameter_m: float = 0.85
    welch_nperseg: int = 2_048
    welch_overlap: int = 1_024
    stft_nperseg: int = 1_024
    stft_overlap: int = 512
    fixed_bands_hz: tuple[tuple[float, float], ...] = (
        (0.0, 50.0),
        (50.0, 100.0),
        (100.0, 200.0),
        (200.0, 400.0),
        (400.0, 800.0),
        (800.0, 1_600.0),
        (1_600.0, 3_200.0),
        (3_200.0, 5_001.0),
    )
    wavelength_bands_m: tuple[tuple[float, float], ...] = (
        (0.02, 0.04),
        (0.04, 0.08),
        (0.08, 0.16),
        (0.16, 0.32),
        (0.32, 0.64),
    )
    spatial_step_m: float = 0.005
    spatial_min_transitions: int = 20
    spatial_welch_nperseg: int = 1_024
    spatial_welch_overlap: int = 512
    spatial_window_nperseg: int = 512
    spatial_window_overlap: int = 256
    spatial_feature_families: tuple[str, ...] = (
        "time",
        "spatial_core",
        "spatial_time",
    )
    spatial_linear_c_values: tuple[float, ...] = (0.003, 0.01, 0.03)
    spatial_rbf_c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    spatial_rbf_gamma_values: tuple[str | float, ...] = ("scale",)
    speed_weight_caps: tuple[float, ...] = (1.0, 2.0)
    feature_families: tuple[str, ...] = ("time", "compact")
    c_values: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
    positive_weight_multipliers: tuple[float, ...] = (0.75, 1.0, 1.5)
    include_raw_speed_options: tuple[bool, ...] = (False,)
    side_i_bias_values: tuple[float, ...] = (0.0, 0.2, 0.4)
    outer_folds: int = 5
    outer_repeats: int = 5
    inner_folds: int = 4
    random_state: int = 42
    threshold_grid_size: int = 41
    minimum_mean_macro_f1: float = 0.70
    minimum_side_i_recall: float = 0.50
    minimum_side_ii_recall: float = 0.60
    minimum_overlap_macro_f1: float = 0.65
