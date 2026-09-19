"""Deterministic Door-cycle segmentation and operation inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from railguard.door.config import (
    CLOSE_COMMAND_COLUMN,
    OPEN_COMMAND_COLUMN,
    POSITION_COLUMN,
    TIME_COLUMN,
)
from railguard.door.parsing import DoorDataError


@dataclass
class DoorSegment:
    """One complete Open or Close cycle."""

    segment_index: int
    start_row: int
    end_row: int
    start_time: str
    end_time: str
    operation: str
    frame: pd.DataFrame
    operation_warning: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class SegmentationDiagnostics:
    cycle_count: int
    median_within_cycle_gap_ms: float
    minimum_boundary_gap_ms: float | None
    maximum_within_cycle_gap_ms: float


def infer_operation(frame: pd.DataFrame) -> tuple[str, str | None]:
    """Infer Open/Close from commands and cross-check with position direction."""

    open_vote = float(frame[OPEN_COMMAND_COLUMN].mean())
    close_vote = float(frame[CLOSE_COMMAND_COLUMN].mean())
    if open_vote == close_vote:
        raise DoorDataError("Cannot infer operation: Open and Close command votes are tied")
    operation = "Open" if open_vote > close_vote else "Close"

    position_delta = float(frame[POSITION_COLUMN].iloc[-1] - frame[POSITION_COLUMN].iloc[0])
    if abs(position_delta) < 1e-9:
        return operation, "Door position did not change during the detected cycle"
    position_operation = "Open" if position_delta > 0 else "Close"
    warning = None
    if position_operation != operation:
        warning = (
            f"Command implies {operation}, but door-position direction implies {position_operation}"
        )
    return operation, warning


def segment_stream(
    frame: pd.DataFrame,
    *,
    gap_threshold_ms: float = 1_000.0,
) -> tuple[list[DoorSegment], SegmentationDiagnostics]:
    """Split a validated stream at timestamp gaps larger than the threshold."""

    if "_parsed_time" not in frame.columns:
        raise DoorDataError("Door frame must be loaded with load_door_csv before segmentation")
    if gap_threshold_ms <= 0:
        raise ValueError("gap_threshold_ms must be positive")

    gaps_ms = frame["_parsed_time"].diff().dt.total_seconds().mul(1_000.0)
    boundary_rows = np.flatnonzero(gaps_ms.to_numpy() > gap_threshold_ms)
    starts = np.concatenate(([0], boundary_rows))
    ends = np.concatenate((boundary_rows - 1, [len(frame) - 1]))

    segments: list[DoorSegment] = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        cycle = frame.iloc[int(start) : int(end) + 1].copy()
        if len(cycle) < 2:
            raise DoorDataError(f"Detected cycle {index} contains fewer than two readings")
        operation, warning = infer_operation(cycle)
        segments.append(
            DoorSegment(
                segment_index=index,
                start_row=int(start),
                end_row=int(end),
                start_time=str(cycle[TIME_COLUMN].iloc[0]),
                end_time=str(cycle[TIME_COLUMN].iloc[-1]),
                operation=operation,
                frame=cycle,
                operation_warning=warning,
            )
        )

    boundary_gap_values = gaps_ms.iloc[boundary_rows].dropna()
    within_mask = np.ones(len(frame), dtype=bool)
    within_mask[0] = False
    within_mask[boundary_rows] = False
    within_values = gaps_ms.iloc[np.flatnonzero(within_mask)].dropna()
    diagnostics = SegmentationDiagnostics(
        cycle_count=len(segments),
        median_within_cycle_gap_ms=float(within_values.median()),
        minimum_boundary_gap_ms=(
            float(boundary_gap_values.min()) if not boundary_gap_values.empty else None
        ),
        maximum_within_cycle_gap_ms=float(within_values.max()),
    )
    return segments, diagnostics
