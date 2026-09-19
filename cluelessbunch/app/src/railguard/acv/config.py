"""Configuration and schema constants for ACV fault localisation."""

from __future__ import annotations

from dataclasses import dataclass

OUTPUT_COLUMNS = ("file_id", "ranked_cars")

PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "indoor": (
        "Indoor Average Temperature",
        "Passenger Cabin Temperature Detected Value",
    ),
    "outdoor": (
        "Outdoor Average Temperature",
        "Outside Temperature Sensor Reading",
        "Fresh Air Temperature Detected Value",
    ),
    "setpoint": (
        "ACV Control Temperature (Cooling)",
        "Target Temperature Value",
    ),
    "running": ("ACV Running Mode",),
    "valid": ("ACV Information Valid",),
    "system_1_high_pressure": ("Refrigeration System 1 High Pressure Value",),
    "system_1_low_pressure": ("Refrigeration System 1 Low Pressure Value",),
    "system_2_high_pressure": ("Refrigeration System 2 High Pressure Value",),
    "system_2_low_pressure": ("Refrigeration System 2 Low Pressure Value",),
}

CORE_FEATURES = (
    "hottest_fraction",
    "above_setpoint_fraction",
    "peer_temp_q90",
    "control_error_q90",
)


@dataclass(frozen=True)
class AcvConfig:
    """Bounded configuration shared by ACV training and inference."""

    setpoint_exceedance_c: float = 1.0
    temperature_min_c: float = -20.0
    temperature_max_c: float = 80.0
    pressure_min: float = 0.0
    pressure_max: float = 10_000.0
    minimum_peer_cars: int = 3
    minimum_usable_samples: int = 50
    temporal_blocks: int = 4
    dropout_fractions: tuple[float, ...] = (0.1, 0.2, 0.4)
    dropout_repeats: int = 3
    linear_l2_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    learned_min_score_gain: float = 0.02
    confidence_margin: float = 0.05
    random_state: int = 42
