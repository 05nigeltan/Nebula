"""Compact within-car side contrasts, without learning from labels or speed."""

import numpy as np

from railguard.corrugation.parsing import CorrugationSignal


def is_car_feature(name: str) -> bool:
    return name.startswith(("side_i_car_rms_", "side_ii_car_rms_"))


def car_contrast_features(signal: CorrugationSignal) -> dict[str, float]:
    """Add nine measurements per side using matched car-level RMS summaries.

    Contrasts are (own-other)/(own+other), with zero for two silent sides.
    Aggregation is invariant to car numbering and does not assume temporal alignment
    between different cars. Sensor gains specific to one side remain a limitation.
    """
    values = np.asarray(signal.signals, dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=0)).astype(float)
    contrasts = {}
    for kind in ("vibration", "shock"):
        medians = np.zeros((8, 2))
        for car in range(1, 9):
            for index, side in enumerate(("side_i", "side_ii")):
                indices = [
                    c.column_index
                    for c in signal.channels
                    if c.car == car and c.side == side and c.signal_type == kind
                ]
                if len(indices) != 4:
                    raise ValueError(f"Expected four {kind} channels for car {car}, {side}")
                medians[car - 1, index] = np.median(rms[indices])
        total = medians.sum(axis=1)
        contrasts[kind] = np.divide(
            medians[:, 0] - medians[:, 1], total, out=np.zeros(8), where=total > 0
        )
    result = {}
    for side, direction in (("side_i", 1), ("side_ii", -1)):
        for kind, contrast in contrasts.items():
            own = direction * contrast
            prefix = f"{side}_car_rms_{kind}_"
            result.update(
                {
                    prefix + "mean": float(own.mean()),
                    prefix + "std": float(own.std()),
                    prefix + "positive_fraction": float(np.mean(own > 0)),
                    prefix + "top2_mean": float(np.sort(own)[-2:].mean()),
                }
            )
        result[f"{side}_car_rms_joint_positive_fraction"] = float(
            np.mean((direction * contrasts["vibration"] > 0) & (direction * contrasts["shock"] > 0))
        )
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Car features must be finite")
    return result
