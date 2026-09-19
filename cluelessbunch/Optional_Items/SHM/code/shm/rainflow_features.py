"""Deterministic physical and compact statistical features for one SHM file."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import rainflow
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from railguard.shm.config import ShmConfig
from railguard.shm.parsing import ShmSignal


def _safe_share(values: np.ndarray, total: float) -> np.ndarray:
    return values / total if total > 0 else np.zeros_like(values, dtype=float)


def _spectral_features(values: np.ndarray) -> dict[str, float]:
    frequencies, power = welch(values, fs=1.0, nperseg=min(4096, len(values)))
    total = float(np.trapezoid(power, frequencies))
    centroid = float(np.sum(frequencies * power) / np.sum(power)) if np.sum(power) > 0 else 0.0
    result = {"stat_spectral_centroid": centroid}
    edges = (0.0, 0.05, 0.15, 0.30, 0.5 + np.finfo(float).eps)
    for index, (lower, upper) in enumerate(pairwise(edges)):
        mask = (frequencies >= lower) & (frequencies < upper)
        band = float(np.trapezoid(power[mask], frequencies[mask])) if np.sum(mask) > 1 else 0.0
        result[f"stat_band_{index}_share"] = band / total if total > 0 else 0.0
    return result


def _top_damage_share(damage: np.ndarray, fraction: float) -> float:
    if not len(damage):
        return 0.0
    count = max(1, int(np.ceil(len(damage) * fraction)))
    total = float(np.sum(damage))
    return float(np.sum(np.partition(damage, -count)[-count:]) / total) if total > 0 else 0.0


def extract_features(signal: ShmSignal, config: ShmConfig) -> dict[str, float | int | str]:
    """Extract whole-file features; no target values are used."""

    values = signal.values
    extracted = list(rainflow.extract_cycles(values))
    if not extracted:
        raise ValueError(f"No rainflow cycles could be extracted from {signal.file_id}")
    ranges = np.asarray([cycle[0] for cycle in extracted], dtype=float)
    means = np.asarray([cycle[1] for cycle in extracted], dtype=float)
    counts = np.asarray([cycle[2] for cycle in extracted], dtype=float)
    starts = np.asarray([cycle[3] for cycle in extracted], dtype=float)
    ends = np.asarray([cycle[4] for cycle in extracted], dtype=float)
    amplitudes = ranges / 2.0
    damage5 = counts * amplitudes**5
    total_count = float(np.sum(counts))
    total_damage5 = float(np.sum(damage5))

    features: dict[str, float | int | str] = {
        "file_id": signal.file_id,
        "sha256": signal.sha256,
        "sample_count": signal.sample_count,
        "rf_extracted_cycles": len(extracted),
        "rf_weighted_cycle_count": total_count,
        "rf_half_cycle_entries": int(np.sum(np.isclose(counts, 0.5))),
        "rf_full_cycle_entries": int(np.sum(np.isclose(counts, 1.0))),
        "rf_max_range": float(np.max(ranges)),
        "rf_weighted_mean_range": float(np.average(ranges, weights=counts)),
        "rf_weighted_mean_cycle_mean": float(np.average(means, weights=counts)),
        "rf_weighted_mean_abs_cycle_mean": float(np.average(np.abs(means), weights=counts)),
        "rf_top_01_damage_share": _top_damage_share(damage5, 0.01),
        "rf_top_05_damage_share": _top_damage_share(damage5, 0.05),
        "rf_top_10_damage_share": _top_damage_share(damage5, 0.10),
        "stat_mean": float(np.mean(values)),
        "stat_std": float(np.std(values)),
        "stat_rms": float(np.sqrt(np.mean(values**2))),
        "stat_skew": float(skew(values, bias=False)),
        "stat_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
        "stat_range": float(np.ptp(values)),
        "stat_diff_energy": float(np.mean(np.diff(values) ** 2)),
    }
    for exponent in config.exponents:
        physical_sum = float(np.sum(counts * amplitudes**exponent))
        features[f"log_b_m_{exponent:g}"] = float(np.log(max(physical_sum, config.positive_floor)))
    for quantile in (0.50, 0.75, 0.90, 0.95, 0.99):
        features[f"stat_abs_q{int(quantile * 100):02d}"] = float(
            np.quantile(np.abs(values), quantile)
        )

    bin_edges = np.asarray((*config.amplitude_bins, np.inf), dtype=float)
    bin_indices = np.clip(
        np.digitize(amplitudes, bin_edges, right=False) - 1, 0, len(bin_edges) - 2
    )
    count_by_bin = np.bincount(bin_indices, weights=counts, minlength=len(bin_edges) - 1)
    damage_by_bin = np.bincount(bin_indices, weights=damage5, minlength=len(bin_edges) - 1)
    for index, value in enumerate(_safe_share(count_by_bin, total_count)):
        features[f"rf_bin_{index}_count_share"] = float(value)
    for index, value in enumerate(_safe_share(damage_by_bin, total_damage5)):
        features[f"rf_bin_{index}_damage_share"] = float(value)

    cycle_positions = (starts + ends) / (2.0 * max(len(values) - 1, 1))
    quarter_indices = np.clip((cycle_positions * 4).astype(int), 0, 3)
    quarter_damage = np.bincount(quarter_indices, weights=damage5, minlength=4)
    quarter_counts = np.bincount(quarter_indices, weights=counts, minlength=4)
    for index, value in enumerate(_safe_share(quarter_damage, total_damage5)):
        features[f"rf_quarter_{index}_damage_share"] = float(value)
    for index, value in enumerate(_safe_share(quarter_counts, total_count)):
        features[f"rf_quarter_{index}_count_share"] = float(value)

    features.update(_spectral_features(values))
    numeric = [value for key, value in features.items() if key not in {"file_id", "sha256"}]
    if not np.all(np.isfinite(np.asarray(numeric, dtype=float))):
        raise ValueError(f"Feature extraction produced a non-finite value for {signal.file_id}")
    return features


def feature_columns_for_group(columns: Iterable[str], group: str) -> list[str]:
    """Select compact residual features without the primary physical basis itself."""

    columns = list(columns)
    spectrum = [name for name in columns if name.startswith(("rf_bin_", "rf_top_"))]
    mean_and_chronology = [
        name for name in columns if "cycle_mean" in name or name.startswith("rf_quarter_")
    ]
    compact_stats = [name for name in columns if name.startswith("stat_")]
    common = [
        "rf_weighted_cycle_count",
        "rf_max_range",
        "rf_weighted_mean_range",
        "rf_half_cycle_entries",
    ]
    if group == "spectrum":
        selected = spectrum
    elif group == "spectrum_chronology":
        selected = spectrum + mean_and_chronology
    elif group == "compact_all":
        selected = spectrum + mean_and_chronology + compact_stats + common
    else:
        raise ValueError(f"Unknown SHM feature group {group!r}")
    return sorted({name for name in selected if name in columns})
