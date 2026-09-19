"""Configuration and schema constants for the Door subsystem."""

from __future__ import annotations

from dataclasses import dataclass

TIME_COLUMN = "Datetime"
CURRENT_COLUMN = "Motor current(mA)"
VOLTAGE_COLUMN = "Motor Voltage(10mV)"
BACK_EMF_COLUMN = "Motor electrodynamic force"
POSITION_COLUMN = "Door leaf position"
OPEN_COMMAND_COLUMN = "Open command"
CLOSE_COMMAND_COLUMN = "Close command"
OPENING_COLUMN = "Door is opening"
CLOSING_COLUMN = "Door is closing"

REQUIRED_COLUMNS = (
    TIME_COLUMN,
    CURRENT_COLUMN,
    VOLTAGE_COLUMN,
    BACK_EMF_COLUMN,
    "Door opening time(.1s)",
    "Door closing time(.1s)",
    CLOSE_COMMAND_COLUMN,
    OPEN_COMMAND_COLUMN,
    "DCSR",
    "DCSL",
    "DLSR",
    "DLSL",
    "Door Opened",
    "Door Locked",
    OPENING_COLUMN,
    CLOSING_COLUMN,
    POSITION_COLUMN,
)

NUMERIC_COLUMNS = tuple(column for column in REQUIRED_COLUMNS if column != TIME_COLUMN)

NORMAL_LABEL = "Normal"
ABNORMAL_LABEL = "Abnormal resistance"
VALID_LABELS = (NORMAL_LABEL, ABNORMAL_LABEL)
OUTPUT_COLUMNS = ("start_time", "end_time", "prediction")


@dataclass(frozen=True)
class DoorConfig:
    """Runtime configuration shared by training and inference."""

    gap_threshold_ms: float = 1_000.0
    template_points: int = 64
    outer_folds: int = 5
    inner_folds: int = 4
    random_state: int = 42
