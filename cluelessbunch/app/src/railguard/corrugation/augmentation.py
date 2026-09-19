"""Topology-preserving training views for rail-corrugation recordings."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.features import (
    aggregate_channel_features,
    compute_channel_features,
)
from railguard.corrugation.parsing import CorrugationSignal

ALL_CARS = frozenset(range(1, 9))
VIEW_COUNT = 9
VIEW_METADATA_COLUMNS = (
    "source_file_id",
    "view_id",
    "omitted_car",
    "view_weight",
)


def extract_jackknife_views(
    signal: CorrugationSignal,
    config: CorrugationConfig,
) -> list[dict[str, object]]:
    """Return the full view and one deterministic leave-one-car-out view per car."""

    channel_features = compute_channel_features(signal, config)
    view_weight = 1.0 / VIEW_COUNT
    rows = [
        aggregate_channel_features(
            channel_features,
            config,
            ALL_CARS,
            {
                "source_file_id": signal.file_id,
                "view_id": "full",
                "omitted_car": 0,
                "view_weight": view_weight,
            },
        )
    ]
    for omitted_car in sorted(ALL_CARS):
        rows.append(
            aggregate_channel_features(
                channel_features,
                config,
                ALL_CARS.difference({omitted_car}),
                {
                    "source_file_id": signal.file_id,
                    "view_id": f"drop_car_{omitted_car}",
                    "omitted_car": omitted_car,
                    "view_weight": view_weight,
                },
            )
        )
    return rows


def full_features_from_views(views: pd.DataFrame) -> pd.DataFrame:
    """Recover one v1-compatible full-sensor feature row per source recording."""

    full = views.loc[views["view_id"] == "full"].copy()
    if full["source_file_id"].duplicated().any():
        raise ValueError("Each source recording must have exactly one full view")
    full = full.drop(columns=list(VIEW_METADATA_COLUMNS))
    return full.reset_index(drop=True)


def select_views(views: pd.DataFrame, source_ids: Iterable[str]) -> pd.DataFrame:
    """Select complete view groups for a set of source files."""

    wanted = set(source_ids)
    selected = views.loc[views["source_file_id"].isin(wanted)].copy()
    observed = set(selected["source_file_id"])
    if observed != wanted:
        missing = sorted(wanted.difference(observed))
        raise ValueError(f"Missing augmented views for source files: {missing}")
    validate_view_groups(selected)
    return selected.reset_index(drop=True)


def validate_view_groups(views: pd.DataFrame) -> None:
    """Check completeness, uniqueness, and weight conservation for view groups."""

    required = {"source_file_id", "view_id", "omitted_car", "view_weight"}
    missing = required.difference(views.columns)
    if missing:
        raise ValueError(f"Augmented feature frame is missing columns: {sorted(missing)}")
    expected_ids = {"full", *(f"drop_car_{car}" for car in sorted(ALL_CARS))}
    for source_id, group in views.groupby("source_file_id", sort=False):
        if len(group) != VIEW_COUNT or set(group["view_id"]) != expected_ids:
            raise ValueError(f"Incomplete or duplicate sensor views for {source_id}")
        if not np.isclose(group["view_weight"].sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"View weights do not sum to one for {source_id}")
