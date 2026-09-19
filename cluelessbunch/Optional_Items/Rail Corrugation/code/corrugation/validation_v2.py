"""Uncertainty and speed-shift checks for the experimental corrugation v2 model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.config import (
    NORMAL_LABEL,
    SIDE_I_LABEL,
    SIDE_II_LABEL,
    CorrugationConfig,
)
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import choose_threshold
from railguard.corrugation.models_v2 import FittedCorrugationV2, V2ModelSpec
from railguard.corrugation.training import duplicate_groups


def paired_file_bootstrap(
    predictions: pd.DataFrame,
    iterations: int = 2_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap unique files within each repeat and compare macro-F1."""

    rng = np.random.default_rng(random_state)
    deltas = []
    for _, repeat_frame in predictions.groupby("repeat", sort=True):
        repeat_frame = repeat_frame.reset_index(drop=True)
        for _ in range(iterations // predictions["repeat"].nunique()):
            sampled = rng.integers(0, len(repeat_frame), len(repeat_frame))
            sample = repeat_frame.iloc[sampled]
            baseline = score_corrugation(
                sample["truth"].to_numpy(), sample["baseline_prediction"].to_numpy()
            ).macro_f1
            v2 = score_corrugation(
                sample["truth"].to_numpy(), sample["v2_prediction"].to_numpy()
            ).macro_f1
            deltas.append(v2 - baseline)
    values = np.asarray(deltas, dtype=float)
    return {
        "iterations": len(values),
        "median_delta": float(np.median(values)),
        "p05_delta": float(np.quantile(values, 0.05)),
        "p95_delta": float(np.quantile(values, 0.95)),
        "probability_v2_better": float(np.mean(values > 0)),
    }


def per_repeat_comparison(predictions: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        baseline = score_corrugation(
            frame["truth"].to_numpy(), frame["baseline_prediction"].to_numpy()
        ).macro_f1
        v2 = score_corrugation(
            frame["truth"].to_numpy(), frame["v2_prediction"].to_numpy()
        ).macro_f1
        rows.append(
            {
                "repeat": int(repeat),
                "baseline_macro_f1": baseline,
                "v2_macro_f1": v2,
                "delta": v2 - baseline,
            }
        )
    return rows


def _threshold_for_fixed_v2(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    spec: V2ModelSpec,
    config: CorrugationConfig,
) -> V2ModelSpec:
    first_scores = np.empty(len(labels), dtype=float)
    second_scores = np.empty(len(labels), dtype=float)
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    raw_spec = replace(spec, threshold=0.0, side_i_bias=0.0)
    for train, validation in splitter.split(np.zeros(len(labels)), labels, groups):
        model = FittedCorrugationV2(raw_spec, config.random_state).fit(
            features.iloc[train], labels[train]
        )
        first, second = model.decision_scores(features.iloc[validation])
        first_scores[validation] = first
        second_scores[validation] = second
    threshold, side_i_bias, _ = choose_threshold(
        labels,
        first_scores,
        second_scores,
        config.threshold_grid_size,
        config.minimum_side_i_recall,
        config.minimum_side_ii_recall,
        config.side_i_bias_values,
    )
    return replace(spec, threshold=threshold, side_i_bias=side_i_bias)


def speed_bin_holdouts_v2(
    features: pd.DataFrame,
    labels: np.ndarray,
    spec: V2ModelSpec,
    config: CorrugationConfig,
) -> list[dict[str, Any]]:
    speed = features["speed_mps"].to_numpy(float)
    groups = duplicate_groups(features)
    output = []
    for lower, upper in ((9.5, 12.0), (12.0, 15.0), (15.0, np.inf)):
        validation = np.flatnonzero((speed >= lower) & (speed < upper))
        train = np.flatnonzero(~((speed >= lower) & (speed < upper)))
        tuned = _threshold_for_fixed_v2(
            features.iloc[train].reset_index(drop=True),
            labels[train],
            groups[train],
            spec,
            config,
        )
        model = FittedCorrugationV2(tuned, config.random_state).fit(
            features.iloc[train], labels[train]
        )
        predicted = model.predict(features.iloc[validation])
        score = score_corrugation(labels[validation], predicted)
        output.append(
            {
                "lower_mps": lower,
                "upper_mps": upper,
                "files": len(validation),
                "class_counts": pd.Series(labels[validation]).value_counts().to_dict(),
                **score.as_dict(),
            }
        )
    return output


def build_v2_validation_report(
    features: pd.DataFrame,
    labels: np.ndarray,
    predictions: pd.DataFrame,
    training_report: dict[str, Any],
    spec: V2ModelSpec,
    config: CorrugationConfig,
) -> dict[str, Any]:
    bootstrap = paired_file_bootstrap(predictions)
    speed_holdouts = speed_bin_holdouts_v2(features, labels, spec, config)
    high_speed = speed_holdouts[-1]
    v2 = training_report["candidate_metrics"]["v2"]
    baseline = training_report["candidate_metrics"]["baseline"]
    gates = {
        "mean_macro_f1_delta": bool(v2["mean_fold_delta"] >= 0.02),
        "side_i_f1": bool(v2["per_class_f1"][SIDE_I_LABEL] >= 0.55),
        "side_i_recall": bool(v2["per_class_recall"][SIDE_I_LABEL] >= 0.60),
        "normal_f1": bool(v2["per_class_f1"][NORMAL_LABEL] >= 0.94),
        "side_ii_guardrail": bool(
            v2["per_class_f1"][SIDE_II_LABEL]
            >= baseline["per_class_f1"][SIDE_II_LABEL] - 0.03
        ),
        "bootstrap_median_delta": bool(bootstrap["median_delta"] > 0),
        "high_speed_macro_f1": bool(high_speed["macro_f1"] >= 0.65),
        "high_speed_side_i_recall": bool(
            high_speed["per_class_recall"][SIDE_I_LABEL] > 0
        ),
    }
    passed = all(gates.values())
    return {
        "test_data_used": False,
        "paired_file_bootstrap": bootstrap,
        "per_repeat_comparison": per_repeat_comparison(predictions),
        "speed_bin_holdouts": speed_holdouts,
        "acceptance": {**gates, "passed": passed},
        "promotion_decision": "promote_v2" if passed else "reject_v2_keep_v1",
    }
