"""Physically grounded time, frequency, wavelength, and side-consensus features."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.parsing import ChannelInfo, CorrugationSignal

EPSILON = 1e-12


@dataclass(frozen=True)
class ChannelFeatureSet:
    """Per-channel measurements retained for physically valid sensor views."""

    file_id: str
    sha256: str
    channels: tuple[ChannelInfo, ...]
    tach_transitions: int
    speed_mps: float
    tach_duty: float
    per_channel: Mapping[str, np.ndarray]
    dominant_wavelength_band: np.ndarray


def tachometer_speed(
    tachometer: np.ndarray,
    config: CorrugationConfig,
) -> tuple[int, float, float]:
    """Return transition count, metres per second, and binary duty cycle."""

    transitions = int(np.count_nonzero(np.diff(np.asarray(tachometer))))
    duration_seconds = len(tachometer) / config.sample_rate_hz
    revolutions = transitions / (2.0 * config.tach_teeth)
    speed = np.pi * config.wheel_diameter_m * revolutions / duration_seconds
    return transitions, float(speed), float(np.mean(tachometer))


def _band_ratios(
    power: np.ndarray,
    frequencies: np.ndarray,
    bands: tuple[tuple[float, float], ...],
) -> np.ndarray:
    total = np.sum(power, axis=0)
    rows = []
    for lower, upper in bands:
        mask = (frequencies >= lower) & (frequencies < upper)
        if np.any(mask):
            rows.append(np.sum(power[mask], axis=0) / np.maximum(total, EPSILON))
        else:
            rows.append(np.zeros(power.shape[1], dtype=float))
    return np.asarray(rows)


def _wavelength_frequency_bands(
    speed_mps: float,
    wavelength_bands: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if speed_mps <= 0:
        return tuple((np.inf, np.inf) for _ in wavelength_bands)
    return tuple(
        (speed_mps / long_wavelength, speed_mps / short_wavelength)
        for short_wavelength, long_wavelength in wavelength_bands
    )


def _windowed_wavelength_variation(
    centered: np.ndarray,
    speed_mps: float,
    config: CorrugationConfig,
) -> np.ndarray:
    """Per-channel variability of wavelength-band energy across short windows."""

    band_count = len(config.wavelength_bands_m)
    if speed_mps <= 0:
        return np.zeros((band_count, centered.shape[1]), dtype=float)
    window_length = min(config.stft_nperseg, len(centered))
    overlap = min(config.stft_overlap, window_length - 1)
    step = max(1, window_length - overlap)
    starts = list(range(0, len(centered) - window_length + 1, step))
    if not starts:
        starts = [0]
    window = np.hanning(window_length).astype(np.float32)
    spectra = []
    for start in starts:
        segment = centered[start : start + window_length] * window[:, None]
        spectra.append(np.abs(np.fft.rfft(segment, axis=0)) ** 2)
    power = np.asarray(spectra)
    frequencies = np.fft.rfftfreq(window_length, d=1.0 / config.sample_rate_hz)
    total = np.sum(power, axis=1)
    variations = []
    for lower, upper in _wavelength_frequency_bands(speed_mps, config.wavelength_bands_m):
        mask = (frequencies >= lower) & (frequencies < upper)
        if not np.any(mask):
            variations.append(np.zeros(centered.shape[1], dtype=float))
            continue
        ratios = np.sum(power[:, mask, :], axis=1) / np.maximum(total, EPSILON)
        variations.append(np.std(ratios, axis=0) / np.maximum(np.mean(ratios, axis=0), 1e-6))
    return np.asarray(variations)


def _aggregate(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_max": float(np.max(values)),
    }


def compute_channel_features(
    signal: CorrugationSignal,
    config: CorrugationConfig,
) -> ChannelFeatureSet:
    """Compute expensive signal features once while retaining channel identity."""

    transitions, speed_mps, tach_duty = tachometer_speed(signal.tachometer, config)
    values = np.asarray(signal.signals, dtype=np.float32)
    centered = values - np.mean(values, axis=0, keepdims=True)

    rms = np.sqrt(np.mean(centered**2, axis=0))
    peak = np.max(np.abs(centered), axis=0)
    crest = peak / np.maximum(rms, EPSILON)
    kurt = np.nan_to_num(kurtosis(centered, axis=0, fisher=True, bias=False))

    nperseg = min(config.welch_nperseg, len(centered))
    overlap = min(config.welch_overlap, nperseg - 1)
    frequencies, power = welch(
        centered,
        fs=config.sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=overlap,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    power[0] = 0
    total_power = np.sum(power, axis=0)
    normalized = power / np.maximum(total_power, EPSILON)
    spectral_entropy = -np.sum(normalized * np.log(np.maximum(normalized, EPSILON)), axis=0)
    spectral_concentration = np.max(normalized, axis=0)
    spectral_centroid = np.sum(frequencies[:, None] * power, axis=0) / np.maximum(
        total_power, EPSILON
    )
    fixed = _band_ratios(power, frequencies, config.fixed_bands_hz)
    wavelength_frequency_bands = _wavelength_frequency_bands(speed_mps, config.wavelength_bands_m)
    wavelength = _band_ratios(power, frequencies, wavelength_frequency_bands)
    wave_variation = _windowed_wavelength_variation(centered, speed_mps, config)

    per_channel: dict[str, np.ndarray] = {
        "rms": rms,
        "peak": peak,
        "crest": crest,
        "kurtosis": kurt,
        "spectral_entropy": spectral_entropy,
        "spectral_concentration": spectral_concentration,
        "spectral_centroid": spectral_centroid,
    }
    for index in range(len(config.fixed_bands_hz)):
        per_channel[f"freq_band_{index}"] = fixed[index]
    for index in range(len(config.wavelength_bands_m)):
        per_channel[f"wave_band_{index}"] = wavelength[index]
        per_channel[f"wave_variation_{index}"] = wave_variation[index]

    dominant = np.argmax(wavelength, axis=0)
    return ChannelFeatureSet(
        file_id=signal.file_id,
        sha256=signal.sha256,
        channels=signal.channels,
        tach_transitions=transitions,
        speed_mps=speed_mps,
        tach_duty=tach_duty,
        per_channel=per_channel,
        dominant_wavelength_band=dominant,
    )


def aggregate_channel_features(
    channel_features: ChannelFeatureSet,
    config: CorrugationConfig,
    included_cars: Collection[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate a topology-preserving car subset into the established feature schema."""

    cars = frozenset(range(1, 9) if included_cars is None else included_cars)
    if not cars or not cars.issubset(range(1, 9)):
        raise ValueError("included_cars must be a non-empty subset of cars 1 through 8")
    row: dict[str, Any] = {
        "file_id": channel_features.file_id,
        "sha256": channel_features.sha256,
        "tach_transitions": channel_features.tach_transitions,
        "speed_mps": channel_features.speed_mps,
        "tach_duty": channel_features.tach_duty,
        "wavelength_valid": float(channel_features.speed_mps > 0),
    }
    if metadata:
        row.update(metadata)
    sides = ("side_i", "side_ii")
    signal_types = ("vibration", "shock")
    for side in sides:
        for signal_type in signal_types:
            selected = np.asarray(
                [
                    channel.car in cars
                    and channel.side == side
                    and channel.signal_type == signal_type
                    for channel in channel_features.channels
                ]
            )
            expected = len(cars) * 4
            if int(np.sum(selected)) != expected:
                raise ValueError(f"Expected {expected} {side} {signal_type} channels")
            for feature_name, feature_values in channel_features.per_channel.items():
                for name, value in _aggregate(
                    feature_values[selected], f"{side}_{signal_type}_{feature_name}"
                ).items():
                    row[name] = value
            if channel_features.speed_mps > 0:
                dominant = channel_features.dominant_wavelength_band[selected]
                for band_index in range(len(config.wavelength_bands_m)):
                    row[f"{side}_{signal_type}_wave_consensus_{band_index}"] = float(
                        np.mean(dominant == band_index)
                    )
            else:
                for band_index in range(len(config.wavelength_bands_m)):
                    row[f"{side}_{signal_type}_wave_consensus_{band_index}"] = 0.0
    return row


def extract_features(
    signal: CorrugationSignal,
    config: CorrugationConfig,
) -> dict[str, Any]:
    """Extract robust full-sensor features using the documented rail-side mapping."""

    return aggregate_channel_features(compute_channel_features(signal, config), config=config)
