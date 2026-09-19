"""Tachometer-resampled wavelength features for rail-corrugation detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.features import EPSILON, extract_features
from railguard.corrugation.parsing import CorrugationSignal


def distance_from_tachometer(
    tachometer: np.ndarray,
    config: CorrugationConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map valid time-sample indices to cumulative wheel travel in metres.

    The documented sensor toggles at both the leading and trailing edge of each
    of 90 teeth, so every transition advances half a tooth pitch.
    """

    tachometer = np.asarray(tachometer)
    edges = np.flatnonzero(np.diff(tachometer) != 0) + 1
    if len(edges) < config.spatial_min_transitions:
        return None
    distance_per_transition = np.pi * config.wheel_diameter_m / (2.0 * config.tach_teeth)
    edge_distance = np.arange(len(edges), dtype=float) * distance_per_transition
    sample_indices = np.arange(edges[0], edges[-1] + 1, dtype=int)
    sample_distance = np.interp(sample_indices, edges, edge_distance)
    if sample_distance[-1] < max(config.wavelength_bands_m)[1] * 2.0:
        return None
    return sample_indices, sample_distance


def resample_to_distance(
    tachometer: np.ndarray,
    signals: np.ndarray,
    config: CorrugationConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate all channels onto an evenly spaced distance grid."""

    mapping = distance_from_tachometer(tachometer, config)
    if mapping is None:
        return None
    sample_indices, sample_distance = mapping
    uniform_distance = np.arange(
        sample_distance[0], sample_distance[-1], config.spatial_step_m, dtype=float
    )
    if len(uniform_distance) < 32:
        return None
    source = np.asarray(signals, dtype=np.float32)[sample_indices]
    resampled = np.empty((len(uniform_distance), source.shape[1]), dtype=np.float32)
    for channel in range(source.shape[1]):
        resampled[:, channel] = np.interp(
            uniform_distance, sample_distance, source[:, channel]
        )
    return uniform_distance, resampled


def _wavelength_masks(
    spatial_frequencies: np.ndarray,
    bands: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[np.ndarray]]:
    lower_wavelength = min(short for short, _ in bands)
    upper_wavelength = max(long for _, long in bands)
    target = (spatial_frequencies >= 1.0 / upper_wavelength) & (
        spatial_frequencies < 1.0 / lower_wavelength
    )
    masks = [
        (spatial_frequencies >= 1.0 / long) & (spatial_frequencies < 1.0 / short)
        for short, long in bands
    ]
    return target, masks


def _window_persistence(
    centered: np.ndarray,
    global_band: np.ndarray,
    config: CorrugationConfig,
) -> np.ndarray:
    window_length = min(config.spatial_window_nperseg, len(centered))
    if window_length < 32:
        return np.zeros(centered.shape[1], dtype=float)
    overlap = min(config.spatial_window_overlap, window_length - 1)
    step = max(1, window_length - overlap)
    starts = list(range(0, len(centered) - window_length + 1, step))
    if not starts:
        starts = [0]
    window = np.hanning(window_length).astype(np.float32)
    frequencies = np.fft.rfftfreq(window_length, d=config.spatial_step_m)
    target, masks = _wavelength_masks(frequencies, config.wavelength_bands_m)
    if not np.any(target):
        return np.zeros(centered.shape[1], dtype=float)
    winners = []
    for start in starts:
        segment = centered[start : start + window_length] * window[:, None]
        power = np.abs(np.fft.rfft(segment, axis=0)) ** 2
        band_power = np.asarray(
            [np.sum(power[mask], axis=0) if np.any(mask) else np.zeros(power.shape[1]) for mask in masks]
        )
        winners.append(np.argmax(band_power, axis=0))
    return np.mean(np.asarray(winners) == global_band[None, :], axis=0)


def channel_spatial_features(
    resampled: np.ndarray,
    config: CorrugationConfig,
) -> dict[str, np.ndarray]:
    """Return one spatial-spectrum feature vector per sensor channel."""

    centered = np.asarray(resampled, dtype=np.float32)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    nperseg = min(config.spatial_welch_nperseg, len(centered))
    overlap = min(config.spatial_welch_overlap, nperseg - 1)
    frequencies, power = welch(
        centered,
        fs=1.0 / config.spatial_step_m,
        window="hann",
        nperseg=nperseg,
        noverlap=overlap,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    target, masks = _wavelength_masks(frequencies, config.wavelength_bands_m)
    if not np.any(target):
        raise ValueError("Spatial sampling does not resolve the configured wavelength range")

    target_power = power[target]
    total = np.sum(target_power, axis=0)
    normalized = target_power / np.maximum(total, EPSILON)
    target_frequencies = frequencies[target]
    peak_index = np.argmax(target_power, axis=0)
    peak_frequency = target_frequencies[peak_index]
    peak_power = np.take_along_axis(target_power, peak_index[None, :], axis=0)[0]
    background = np.median(target_power, axis=0)
    peak_wavelength = 1.0 / np.maximum(peak_frequency, EPSILON)
    peak_prominence = np.log10((peak_power + EPSILON) / (background + EPSILON))
    concentration = peak_power / np.maximum(total, EPSILON)
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, EPSILON)), axis=0)
    entropy /= max(np.log(len(target_frequencies)), EPSILON)

    band_energy = np.asarray(
        [np.sum(power[mask], axis=0) if np.any(mask) else np.zeros(power.shape[1]) for mask in masks]
    )
    band_fraction = band_energy / np.maximum(total, EPSILON)
    dominant_band = np.argmax(band_energy, axis=0)
    persistence = _window_persistence(centered, dominant_band, config)

    features: dict[str, np.ndarray] = {
        "peak_wavelength": peak_wavelength,
        "peak_prominence": peak_prominence,
        "concentration": concentration,
        "entropy": entropy,
        "persistence": persistence,
        "dominant_band": dominant_band.astype(float),
    }
    for index, values in enumerate(band_fraction):
        features[f"band_{index}"] = np.log10(np.maximum(values, EPSILON))
    return features


def _top_k_mean(values: np.ndarray, count: int = 4) -> float:
    count = min(count, len(values))
    return float(np.mean(np.partition(values, len(values) - count)[-count:]))


def _pool_channel_features(
    per_channel: dict[str, np.ndarray],
    selected: np.ndarray,
    prefix: str,
    band_count: int,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for index in range(band_count):
        values = per_channel[f"band_{index}"][selected]
        row[f"{prefix}_band_{index}_q90"] = float(np.quantile(values, 0.90))
        row[f"{prefix}_band_{index}_top4"] = _top_k_mean(values)
    for name in ("peak_prominence", "concentration", "entropy", "persistence"):
        values = per_channel[name][selected]
        row[f"{prefix}_{name}_median"] = float(np.median(values))
        row[f"{prefix}_{name}_q90"] = float(np.quantile(values, 0.90))
    wavelengths = per_channel["peak_wavelength"][selected]
    wavelength_median = float(np.median(wavelengths))
    row[f"{prefix}_peak_wavelength_median"] = wavelength_median
    row[f"{prefix}_peak_wavelength_mad"] = float(
        np.median(np.abs(wavelengths - wavelength_median))
    )
    dominant = per_channel["dominant_band"][selected].astype(int)
    for index in range(band_count):
        row[f"{prefix}_consensus_{index}"] = float(np.mean(dominant == index))
    return row


def extract_spatial_features(
    signal: CorrugationSignal,
    config: CorrugationConfig,
) -> dict[str, Any]:
    """Add compact distance-domain features to the v1 feature record."""

    row = extract_features(signal, config)
    result = resample_to_distance(signal.tachometer, signal.signals, config)
    row["spatial_valid"] = float(result is not None)
    sides = ("side_i", "side_ii")
    signal_types = ("vibration", "shock")
    if result is None:
        template_count = len(config.wavelength_bands_m)
        zeros = {
            **{f"band_{index}": np.zeros(128) for index in range(template_count)},
            "peak_prominence": np.zeros(128),
            "concentration": np.zeros(128),
            "entropy": np.zeros(128),
            "persistence": np.zeros(128),
            "peak_wavelength": np.zeros(128),
            "dominant_band": np.zeros(128),
        }
        per_channel = zeros
    else:
        _, resampled = result
        per_channel = channel_spatial_features(resampled, config)

    for side in sides:
        for signal_type in signal_types:
            selected = np.asarray(
                [
                    channel.side == side and channel.signal_type == signal_type
                    for channel in signal.channels
                ]
            )
            row.update(
                _pool_channel_features(
                    per_channel,
                    selected,
                    f"{side}_{signal_type}_spatial",
                    len(config.wavelength_bands_m),
                )
            )
    numeric = [value for value in row.values() if isinstance(value, (int, float, np.number))]
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Spatial feature extraction produced a non-finite value")
    return row
