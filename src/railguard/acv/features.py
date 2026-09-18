"""Peer-relative, physics-motivated ACV feature extraction."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from railguard.acv.config import CORE_FEATURES, PARAMETER_ALIASES, AcvConfig
from railguard.acv.parsing import AcvCase


def _find_parameter(columns: dict[str, str], kind: str) -> str | None:
    for alias in PARAMETER_ALIASES[kind]:
        if alias in columns:
            return columns[alias]
    return None


def _bounded_numeric(
    frame: pd.DataFrame,
    column: str | None,
    lower: float,
    upper: float,
) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.where(values.gt(lower) & values.le(upper))


def _schema_name(case: AcvCase) -> str:
    parameters = {
        parameter for columns in case.car_columns.values() for parameter in columns
    }
    return (
        "rich_pressure"
        if any("Pressure Value" in parameter for parameter in parameters)
        else "standard"
    )


def _pressure_diagnostics(
    case: AcvCase,
    active: pd.DataFrame,
    valid: pd.DataFrame,
    config: AcvConfig,
) -> dict[str, dict[str, float]]:
    cars = list(case.car_ids)
    values: dict[str, pd.DataFrame] = {}
    for kind in (
        "system_1_high_pressure",
        "system_1_low_pressure",
        "system_2_high_pressure",
        "system_2_low_pressure",
    ):
        matrix = pd.DataFrame(index=case.frame.index)
        for car in cars:
            matrix[car] = _bounded_numeric(
                case.frame,
                _find_parameter(case.car_columns[car], kind),
                config.pressure_min,
                config.pressure_max,
            )
        values[kind] = matrix.where(active & valid)

    result: dict[str, dict[str, float]] = {car: {} for car in cars}
    for kind, matrix in values.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            peer_median = matrix.median(axis=1)
        # A positive deficit means the car pressure is below the contemporaneous peer median.
        deficit = matrix.rsub(peer_median, axis=0)
        for car in cars:
            series = deficit[car].dropna()
            result[car][f"{kind}_deficit_q90"] = (
                float(series.quantile(0.9)) if not series.empty else np.nan
            )
    high_imbalance = (values["system_1_high_pressure"] - values["system_2_high_pressure"]).abs()
    low_imbalance = (values["system_1_low_pressure"] - values["system_2_low_pressure"]).abs()
    for car in cars:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[car]["high_pressure_system_imbalance_median"] = float(
                high_imbalance[car].median()
            )
            result[car]["low_pressure_system_imbalance_median"] = float(
                low_imbalance[car].median()
            )
    return result


def add_within_case_ranks(features: pd.DataFrame) -> pd.DataFrame:
    """Rank higher raw values as more fault-like, leaving missing evidence at zero."""

    ranked = features.copy()
    for feature in CORE_FEATURES:
        rank_column = f"rank_{feature}"
        ranked[rank_column] = ranked.groupby("file_id")[feature].transform(
            lambda values: values.rank(method="average", pct=True)
        )
        ranked[rank_column] = ranked[rank_column].fillna(0.0)
    return ranked


def extract_case_features(case: AcvCase, config: AcvConfig) -> pd.DataFrame:
    """Return one physics-informed feature record per car in a complete case."""

    cars = list(case.car_ids)
    indoor = pd.DataFrame(index=case.frame.index)
    outdoor = pd.DataFrame(index=case.frame.index)
    setpoint = pd.DataFrame(index=case.frame.index)
    active = pd.DataFrame(False, index=case.frame.index, columns=cars)
    valid = pd.DataFrame(True, index=case.frame.index, columns=cars)

    for car in cars:
        columns = case.car_columns[car]
        indoor[car] = _bounded_numeric(
            case.frame,
            _find_parameter(columns, "indoor"),
            config.temperature_min_c,
            config.temperature_max_c,
        )
        outdoor[car] = _bounded_numeric(
            case.frame,
            _find_parameter(columns, "outdoor"),
            config.temperature_min_c,
            config.temperature_max_c,
        )
        setpoint[car] = _bounded_numeric(
            case.frame,
            _find_parameter(columns, "setpoint"),
            config.temperature_min_c,
            config.temperature_max_c,
        )
        running_column = _find_parameter(columns, "running")
        if running_column is None:
            active[car] = indoor[car].notna() & setpoint[car].notna()
        else:
            active[car] = case.frame[running_column].astype("string").str.contains(
                "Cooling", case=False, na=False
            )
        valid_column = _find_parameter(columns, "valid")
        if valid_column is not None:
            valid[car] = case.frame[valid_column].astype("string").str.casefold().eq("valid")

    indoor = indoor.where(valid)
    outdoor = outdoor.where(valid)
    setpoint = setpoint.where(valid)
    cooling = active & indoor.notna() & setpoint.notna()
    active_indoor = indoor.where(active)
    peer_count = active_indoor.notna().sum(axis=1)
    peer_median = active_indoor.median(axis=1)
    peer_maximum = active_indoor.max(axis=1)
    peer_residual = active_indoor.sub(peer_median, axis=0)
    control_error = indoor - setpoint
    schema = _schema_name(case)
    pressure = _pressure_diagnostics(case, active, valid, config) if schema == "rich_pressure" else {}

    rows: list[dict[str, float | int | str | bool]] = []
    for car in cars:
        mask = cooling[car] & peer_count.ge(config.minimum_peer_cars)
        residual = peer_residual.loc[mask, car].dropna()
        error = control_error.loc[mask, car].dropna()
        hottest = indoor.loc[mask, car].eq(peer_maximum.loc[mask]).astype(float)
        ambient_gap = (indoor.loc[mask, car] - outdoor.loc[mask, car]).dropna()
        usable_samples = int(mask.sum())
        row: dict[str, float | int | str | bool] = {
            "file_id": case.file_id,
            "car": car,
            "schema": schema,
            "sample_count": case.sample_count,
            "usable_samples": usable_samples,
            "usable_fraction": float(mask.mean()),
            "supported": usable_samples >= config.minimum_usable_samples,
            "hottest_fraction": float(hottest.mean()) if not hottest.empty else np.nan,
            "above_setpoint_fraction": (
                float((error > config.setpoint_exceedance_c).mean())
                if not error.empty
                else np.nan
            ),
            "peer_temp_median": float(residual.median()) if not residual.empty else np.nan,
            "peer_temp_q90": float(residual.quantile(0.9)) if not residual.empty else np.nan,
            "control_error_median": float(error.median()) if not error.empty else np.nan,
            "control_error_q90": float(error.quantile(0.9)) if not error.empty else np.nan,
            "indoor_minus_outdoor_median": (
                float(ambient_gap.median()) if not ambient_gap.empty else np.nan
            ),
            "cooling_fraction": float(active[car].mean()),
            "invalid_fraction": float((~valid[car]).mean()),
        }
        row.update(pressure.get(car, {}))
        rows.append(row)
    return add_within_case_ranks(pd.DataFrame(rows))
