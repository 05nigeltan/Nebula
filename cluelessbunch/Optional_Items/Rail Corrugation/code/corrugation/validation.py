"""Train-only robustness checks for the selected corrugation model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.config import SIDE_I_LABEL, SIDE_II_LABEL, CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import (
    FittedCorrugationModel,
    ModelSpec,
    choose_threshold,
)
from railguard.corrugation.training import duplicate_groups, nested_validate


def _threshold_for_fixed_spec(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    spec: ModelSpec,
    config: CorrugationConfig,
) -> ModelSpec:
    first_scores = np.empty(len(labels), dtype=float)
    second_scores = np.empty(len(labels), dtype=float)
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    raw_spec = replace(spec, threshold=0.0)
    for train, validation in splitter.split(np.zeros(len(labels)), labels, groups):
        model = FittedCorrugationModel(raw_spec, config.random_state).fit(
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


def repeated_fixed_spec_validation(
    features: pd.DataFrame,
    labels: np.ndarray,
    spec: ModelSpec,
    config: CorrugationConfig,
    repeats: int = 3,
) -> dict[str, Any]:
    groups = duplicate_groups(features)
    truth: list[str] = []
    predictions: list[str] = []
    fold_scores = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state + 100 + repeat,
        )
        for train, validation in splitter.split(np.zeros(len(labels)), labels, groups):
            tuned = _threshold_for_fixed_spec(
                features.iloc[train].reset_index(drop=True),
                labels[train],
                groups[train],
                spec,
                config,
            )
            model = FittedCorrugationModel(tuned, config.random_state + repeat).fit(
                features.iloc[train], labels[train]
            )
            predicted = model.predict(features.iloc[validation])
            truth.extend(labels[validation])
            predictions.extend(predicted)
            fold_scores.append(score_corrugation(labels[validation], predicted).macro_f1)
    score = score_corrugation(np.asarray(truth), np.asarray(predictions))
    return {
        **score.as_dict(),
        "fold_macro_f1_mean": float(np.mean(fold_scores)),
        "fold_macro_f1_std": float(np.std(fold_scores)),
        "fold_macro_f1_min": float(np.min(fold_scores)),
    }


def speed_bin_holdouts(
    features: pd.DataFrame,
    labels: np.ndarray,
    spec: ModelSpec,
    config: CorrugationConfig,
) -> list[dict[str, Any]]:
    speed = features["speed_mps"].to_numpy(float)
    groups = duplicate_groups(features)
    bins = ((9.5, 12.0), (12.0, 15.0), (15.0, np.inf))
    output = []
    for lower, upper in bins:
        validation = np.flatnonzero((speed >= lower) & (speed < upper))
        train = np.flatnonzero(~((speed >= lower) & (speed < upper)))
        tuned = _threshold_for_fixed_spec(
            features.iloc[train].reset_index(drop=True),
            labels[train],
            groups[train],
            spec,
            config,
        )
        model = FittedCorrugationModel(tuned, config.random_state).fit(
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


def run_robustness_suite(
    features: pd.DataFrame,
    labels: np.ndarray,
    final_spec: ModelSpec,
    config: CorrugationConfig,
) -> dict[str, Any]:
    overlap = features["speed_mps"].to_numpy(float) >= 9.5
    overlap_config = replace(config, outer_repeats=min(3, config.outer_repeats))
    overlap_nested = nested_validate(
        features.loc[overlap].reset_index(drop=True), labels[overlap], overlap_config
    )
    overlap_score = overlap_nested["candidate_scores"]["side_symmetric_linear_svm"]

    ablations = {}
    for name, frame in {
        "all_signals": features,
        "vibration_only": features.drop(
            columns=[column for column in features if "_shock_" in column]
        ),
        "shock_only": features.drop(
            columns=[column for column in features if "_vibration_" in column]
        ),
    }.items():
        ablations[name] = repeated_fixed_spec_validation(
            frame, labels, final_spec, config, repeats=3
        )

    no_raw_speed_spec = replace(final_spec, include_raw_speed=False)
    no_raw_speed = repeated_fixed_spec_validation(
        features, labels, no_raw_speed_spec, config, repeats=3
    )
    overlap_passed = overlap_score["fold_macro_f1_mean"] >= config.minimum_overlap_macro_f1
    return {
        "test_data_used": False,
        "overlap_ge_9_5_mps": {
            "files": int(np.sum(overlap)),
            "class_counts": pd.Series(labels[overlap]).value_counts().to_dict(),
            **overlap_score,
        },
        "signal_type_ablation": ablations,
        "no_raw_speed": no_raw_speed,
        "speed_bin_holdouts": speed_bin_holdouts(features, labels, final_spec, config),
        "acceptance": {
            "overlap_macro_f1": bool(overlap_passed),
            "side_i_recall": bool(overlap_score["per_class_recall"][SIDE_I_LABEL] > 0),
            "side_ii_recall": bool(overlap_score["per_class_recall"][SIDE_II_LABEL] > 0),
            "passed": bool(
                overlap_passed
                and overlap_score["per_class_recall"][SIDE_I_LABEL] > 0
                and overlap_score["per_class_recall"][SIDE_II_LABEL] > 0
            ),
        },
    }
