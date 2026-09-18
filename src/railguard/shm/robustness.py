"""Additional train-only stress tests for the physics and hybrid SHM candidates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from railguard.shm.config import ShmConfig
from railguard.shm.metric import score_shm
from railguard.shm.models import FittedDamageModel, PhysicsSpec, ResidualSpec
from railguard.shm.validation import select_physics_spec, select_residual_spec


def _fit_pair(
    train_frame: pd.DataFrame,
    train_target: np.ndarray,
    validation_frame: pd.DataFrame,
    config: ShmConfig,
) -> tuple[np.ndarray, np.ndarray, PhysicsSpec, ResidualSpec]:
    physics_spec, _ = select_physics_spec(train_frame, train_target, config)
    residual_spec, _ = select_residual_spec(train_frame, train_target, physics_spec, config)
    physics = FittedDamageModel(physics_spec, ResidualSpec("none"), config).fit(
        train_frame, train_target
    )
    hybrid = FittedDamageModel(physics_spec, residual_spec, config).fit(train_frame, train_target)
    return (
        physics.predict(validation_frame),
        hybrid.predict(validation_frame),
        physics_spec,
        residual_spec,
    )


def repeated_stratified_validation(
    frame: pd.DataFrame,
    target: np.ndarray,
    config: ShmConfig,
    repeats: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Repeated eight-fold comparison, stratified by target quartile."""

    strata = pd.qcut(target, q=4, labels=False, duplicates="drop")
    splitter = RepeatedStratifiedKFold(
        n_splits=8,
        n_repeats=repeats,
        random_state=config.random_state,
    )
    records = []
    repeat_metrics = []
    for split_number, (train_index, validation_index) in enumerate(splitter.split(frame, strata)):
        repeat = split_number // 8
        fold = split_number % 8
        physics, hybrid, physics_spec, residual_spec = _fit_pair(
            frame.iloc[train_index].reset_index(drop=True),
            target[train_index],
            frame.iloc[validation_index].reset_index(drop=True),
            config,
        )
        for offset, sample_index in enumerate(validation_index):
            records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "file_id": str(frame.iloc[sample_index]["file_id"]),
                    "truth": target[sample_index],
                    "physics_prediction": physics[offset],
                    "hybrid_prediction": hybrid[offset],
                    "physics_exponent": physics_spec.exponent,
                    "calibration_mode": physics_spec.calibration_mode,
                    "residual_model": residual_spec.name,
                }
            )
    predictions = pd.DataFrame(records)
    for repeat, group in predictions.groupby("repeat"):
        truth = group["truth"].to_numpy(float)
        physics = score_shm(truth, group["physics_prediction"].to_numpy(float))
        hybrid = score_shm(truth, group["hybrid_prediction"].to_numpy(float))
        repeat_metrics.append(
            {
                "repeat": int(repeat),
                "physics_mape": physics.mape,
                "hybrid_mape": hybrid.mape,
                "relative_improvement": (physics.mape - hybrid.mape) / physics.mape,
            }
        )
    physics_ape = (
        np.abs(predictions["physics_prediction"] - predictions["truth"]) / predictions["truth"]
    )
    hybrid_ape = (
        np.abs(predictions["hybrid_prediction"] - predictions["truth"]) / predictions["truth"]
    )
    physics_metrics = score_shm(
        predictions["truth"].to_numpy(float), predictions["physics_prediction"].to_numpy(float)
    )
    hybrid_metrics = score_shm(
        predictions["truth"].to_numpy(float), predictions["hybrid_prediction"].to_numpy(float)
    )
    summary = {
        "repeats": repeats,
        "physics": physics_metrics.as_dict(),
        "hybrid": hybrid_metrics.as_dict(),
        "relative_mape_improvement": (physics_metrics.mape - hybrid_metrics.mape)
        / physics_metrics.mape,
        "sample_fold_win_rate": float(np.mean(hybrid_ape < physics_ape)),
        "repeat_results": repeat_metrics,
        "hybrid_winning_repeats": int(
            sum(item["hybrid_mape"] < item["physics_mape"] for item in repeat_metrics)
        ),
    }
    return summary, predictions


def condition_cluster_validation(
    frame: pd.DataFrame,
    target: np.ndarray,
    config: ShmConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Form label-free operating clusters, then hold each complete cluster out."""

    cluster_columns = [
        "stat_mean",
        "stat_std",
        "stat_rms",
        "stat_skew",
        "stat_kurtosis",
        "stat_diff_energy",
        "stat_spectral_centroid",
    ]
    scaled = StandardScaler().fit_transform(frame[cluster_columns].to_numpy(float))
    clusters = KMeans(n_clusters=3, random_state=config.random_state, n_init=20).fit_predict(scaled)
    records = []
    cluster_results = []
    for cluster in sorted(np.unique(clusters)):
        validation_index = np.flatnonzero(clusters == cluster)
        train_index = np.flatnonzero(clusters != cluster)
        physics, hybrid, physics_spec, residual_spec = _fit_pair(
            frame.iloc[train_index].reset_index(drop=True),
            target[train_index],
            frame.iloc[validation_index].reset_index(drop=True),
            config,
        )
        truth = target[validation_index]
        physics_metrics = score_shm(truth, physics)
        hybrid_metrics = score_shm(truth, hybrid)
        cluster_results.append(
            {
                "cluster": int(cluster),
                "files": len(validation_index),
                "physics_mape": physics_metrics.mape,
                "hybrid_mape": hybrid_metrics.mape,
                "physics_spec": asdict(physics_spec),
                "residual_spec": asdict(residual_spec),
            }
        )
        for offset, sample_index in enumerate(validation_index):
            records.append(
                {
                    "cluster": int(cluster),
                    "file_id": str(frame.iloc[sample_index]["file_id"]),
                    "truth": target[sample_index],
                    "physics_prediction": physics[offset],
                    "hybrid_prediction": hybrid[offset],
                }
            )
    predictions = pd.DataFrame(records)
    summary = {
        "cluster_sizes": {
            str(cluster): int(np.sum(clusters == cluster)) for cluster in np.unique(clusters)
        },
        "physics": score_shm(
            predictions["truth"].to_numpy(float),
            predictions["physics_prediction"].to_numpy(float),
        ).as_dict(),
        "hybrid": score_shm(
            predictions["truth"].to_numpy(float),
            predictions["hybrid_prediction"].to_numpy(float),
        ).as_dict(),
        "clusters": cluster_results,
    }
    return summary, predictions


def extreme_holdouts(
    frame: pd.DataFrame,
    target: np.ndarray,
    config: ShmConfig,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    low, high = np.quantile(target, [0.25, 0.75])
    for name, validation_mask in {
        "lowest_quartile": target <= low,
        "highest_quartile": target >= high,
    }.items():
        validation_index = np.flatnonzero(validation_mask)
        train_index = np.flatnonzero(~validation_mask)
        physics, hybrid, physics_spec, residual_spec = _fit_pair(
            frame.iloc[train_index].reset_index(drop=True),
            target[train_index],
            frame.iloc[validation_index].reset_index(drop=True),
            config,
        )
        results[name] = {
            "files": len(validation_index),
            "target_min": float(np.min(target[validation_index])),
            "target_max": float(np.max(target[validation_index])),
            "physics": score_shm(target[validation_index], physics).as_dict(),
            "hybrid": score_shm(target[validation_index], hybrid).as_dict(),
            "physics_spec": asdict(physics_spec),
            "residual_spec": asdict(residual_spec),
        }
    return results


def paired_bootstrap(
    truth: np.ndarray,
    physics_prediction: np.ndarray,
    hybrid_prediction: np.ndarray,
    random_state: int,
    samples: int = 10_000,
) -> dict[str, float]:
    physics_ape = np.abs(physics_prediction - truth) / truth
    hybrid_ape = np.abs(hybrid_prediction - truth) / truth
    paired_gain = physics_ape - hybrid_ape
    generator = np.random.default_rng(random_state)
    indices = generator.integers(0, len(truth), size=(samples, len(truth)))
    means = np.mean(paired_gain[indices], axis=1)
    return {
        "mean_absolute_mape_gain": float(np.mean(paired_gain)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "probability_gain_positive": float(np.mean(means > 0)),
    }


def residual_ablation(
    frame: pd.DataFrame,
    target: np.ndarray,
    config: ShmConfig,
) -> list[dict[str, Any]]:
    """Compare the three residual feature groups under one fixed regularization setting."""

    physics_spec = PhysicsSpec(5.0, "mape")
    results = []
    indices = np.arange(len(target))
    for group in ("spectrum", "spectrum_chronology", "compact_all"):
        predictions = np.empty(len(target), dtype=float)
        spec = ResidualSpec("ridge", 10.0, 1.0, group)
        for held_out in indices:
            training = indices[indices != held_out]
            model = FittedDamageModel(physics_spec, spec, config).fit(
                frame.iloc[training].reset_index(drop=True), target[training]
            )
            predictions[held_out] = model.predict(frame.iloc[[held_out]].reset_index(drop=True))[0]
        results.append({"feature_group": group, **score_shm(target, predictions).as_dict()})
    return results
