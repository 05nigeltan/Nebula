"""Spatial-feature training and locked comparison for corrugation v2."""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.config import (
    NORMAL_LABEL,
    SIDE_I_LABEL,
    SIDE_II_LABEL,
    CorrugationConfig,
)
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import FittedCorrugationModel, choose_threshold
from railguard.corrugation.models_v2 import (
    FittedCorrugationV2,
    V2ModelSpec,
    v2_feature_bases,
)
from railguard.corrugation.parsing import load_corrugation_file, load_training_manifest, sha256_file
from railguard.corrugation.spatial_features import extract_spatial_features
from railguard.corrugation.training import (
    duplicate_groups,
    feature_cache_signature,
    select_model_spec,
)

FEATURE_SCHEMA_VERSION_V2 = 2


def feature_cache_signature_v2(config: CorrugationConfig) -> dict[str, Any]:
    return {
        **feature_cache_signature(config),
        "schema_version": FEATURE_SCHEMA_VERSION_V2,
        "spatial_step_m": config.spatial_step_m,
        "spatial_min_transitions": config.spatial_min_transitions,
        "spatial_welch_nperseg": config.spatial_welch_nperseg,
        "spatial_welch_overlap": config.spatial_welch_overlap,
        "spatial_window_nperseg": config.spatial_window_nperseg,
        "spatial_window_overlap": config.spatial_window_overlap,
    }


def load_or_extract_training_features_v2(
    train_dir: str | Path,
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    config: CorrugationConfig,
) -> pd.DataFrame:
    train_dir = Path(train_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "train_features.csv"
    metadata_path = cache_dir / "cache_metadata.json"
    signature = feature_cache_signature_v2(config)
    cached_signature = None
    if metadata_path.is_file():
        try:
            cached_signature = json.loads(metadata_path.read_text(encoding="utf-8"))[
                "feature_signature"
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            cached_signature = None
    if cache_path.is_file() and cached_signature == signature:
        cached = pd.read_csv(cache_path)
        ids_match = cached.get("file_id", pd.Series(dtype=str)).tolist() == manifest[
            "filename"
        ].tolist()
        if {"file_id", "sha256", "spatial_valid"}.issubset(cached.columns) and ids_match:
            current_hashes = [sha256_file(train_dir / name) for name in manifest["filename"]]
            if cached["sha256"].tolist() == current_hashes:
                print("Reusing verified corrugation v2 feature cache", flush=True)
                return cached

    rows = []
    for position, filename in enumerate(manifest["filename"], start=1):
        signal = load_corrugation_file(train_dir / filename, config.expected_samples)
        rows.append(extract_spatial_features(signal, config))
        print(
            f"Extracted corrugation v2 features {position:03d}/{len(manifest)}: {filename}",
            flush=True,
        )
    features = pd.DataFrame(rows)
    features.to_csv(cache_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_signature": signature,
                "source_sha256": dict(zip(features["file_id"], features["sha256"], strict=True)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return features


def _candidate_specs(config: CorrugationConfig) -> Iterator[V2ModelSpec]:
    for family in config.spatial_feature_families:
        for speed_cap in config.speed_weight_caps:
            for c_value in config.spatial_linear_c_values:
                yield V2ModelSpec(
                    feature_family=family,
                    kernel="linear",
                    c_value=c_value,
                    speed_weight_cap=speed_cap,
                )
            for c_value in config.spatial_rbf_c_values:
                for gamma in config.spatial_rbf_gamma_values:
                    yield V2ModelSpec(
                        feature_family=family,
                        kernel="rbf",
                        c_value=c_value,
                        gamma=gamma,
                        speed_weight_cap=speed_cap,
                    )


def _inner_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    config: CorrugationConfig,
):
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    yield from splitter.split(np.zeros(len(labels)), labels, groups)


def select_v2_spec(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: CorrugationConfig,
) -> tuple[V2ModelSpec, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_spec: V2ModelSpec | None = None
    family_complexity = {
        "time": 0,
        "spatial_core": 1,
        "spatial_vibration": 2,
        "spatial_time": 3,
        "spatial_all": 4,
    }
    for raw_spec in _candidate_specs(config):
        first_scores = np.empty(len(labels), dtype=float)
        second_scores = np.empty(len(labels), dtype=float)
        for train, validation in _inner_splits(labels, groups, config):
            model = FittedCorrugationV2(raw_spec, config.random_state).fit(
                features.iloc[train], labels[train]
            )
            first, second = model.decision_scores(features.iloc[validation])
            first_scores[validation] = first
            second_scores[validation] = second
        threshold, side_i_bias, score = choose_threshold(
            labels,
            first_scores,
            second_scores,
            config.threshold_grid_size,
            config.minimum_side_i_recall,
            config.minimum_side_ii_recall,
            config.side_i_bias_values,
        )
        selected = replace(raw_spec, threshold=threshold, side_i_bias=side_i_bias)
        minimum_fault_recall = min(
            score.per_class_recall[SIDE_I_LABEL], score.per_class_recall[SIDE_II_LABEL]
        )
        rows.append(
            {
                **asdict(selected),
                "inner_macro_f1": score.macro_f1,
                "inner_side_i_recall": score.per_class_recall[SIDE_I_LABEL],
                "inner_side_ii_recall": score.per_class_recall[SIDE_II_LABEL],
                "side_feature_count": len(v2_feature_bases(features, raw_spec.feature_family)),
            }
        )
        key = (
            score.macro_f1,
            minimum_fault_recall,
            float(raw_spec.kernel == "linear"),
            -float(family_complexity[raw_spec.feature_family]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_spec = selected
    assert best_spec is not None
    rows.sort(key=lambda row: row["inner_macro_f1"], reverse=True)
    return best_spec, rows


def nested_compare(
    features: pd.DataFrame,
    labels: np.ndarray,
    config: CorrugationConfig,
) -> dict[str, Any]:
    """Compare v1 and v2 on identical outer folds with inner-only selection."""

    groups = duplicate_groups(features)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            train_features = features.iloc[train].reset_index(drop=True)
            train_labels = labels[train]
            train_groups = groups[train]

            baseline_spec, _ = select_model_spec(
                train_features, train_labels, train_groups, config
            )
            baseline = FittedCorrugationModel(
                baseline_spec, config.random_state + repeat
            ).fit(features.iloc[train], train_labels)

            v2_spec, _ = select_v2_spec(train_features, train_labels, train_groups, config)
            v2 = FittedCorrugationV2(v2_spec, config.random_state + repeat).fit(
                features.iloc[train], train_labels
            )
            validation_features = features.iloc[validation]
            baseline_prediction = baseline.predict(validation_features)
            v2_first, v2_second = v2.decision_scores(validation_features)
            v2_prediction = v2.predict(validation_features)
            baseline_score = score_corrugation(labels[validation], baseline_prediction)
            v2_score = score_corrugation(labels[validation], v2_prediction)
            fold_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "baseline_macro_f1": baseline_score.macro_f1,
                    "v2_macro_f1": v2_score.macro_f1,
                    "macro_f1_delta": v2_score.macro_f1 - baseline_score.macro_f1,
                    "v2_feature_family": v2_spec.feature_family,
                    "v2_kernel": v2_spec.kernel,
                    "v2_c_value": v2_spec.c_value,
                    "v2_gamma": v2_spec.gamma,
                    "v2_speed_weight_cap": v2_spec.speed_weight_cap,
                    "v2_threshold": v2_spec.threshold,
                    "v2_side_i_bias": v2_spec.side_i_bias,
                }
            )
            validation_set = set(validation.tolist())
            for file_index in range(len(features)):
                split_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": features.iloc[file_index]["file_id"],
                        "role": "validation" if file_index in validation_set else "train",
                    }
                )
            for local, file_index in enumerate(validation):
                prediction_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": features.iloc[file_index]["file_id"],
                        "truth": labels[file_index],
                        "baseline_prediction": baseline_prediction[local],
                        "v2_prediction": v2_prediction[local],
                        "v2_side_i_score": v2_first[local],
                        "v2_side_ii_score": v2_second[local],
                    }
                )
        print(
            f"Completed locked corrugation v2 comparison repeat {repeat + 1}/"
            f"{config.outer_repeats}",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    fold_results = pd.DataFrame(fold_rows)
    baseline_score = score_corrugation(
        predictions["truth"].to_numpy(), predictions["baseline_prediction"].to_numpy()
    )
    v2_score = score_corrugation(
        predictions["truth"].to_numpy(), predictions["v2_prediction"].to_numpy()
    )
    metrics = {
        "baseline": {
            **baseline_score.as_dict(),
            "fold_macro_f1_mean": float(fold_results["baseline_macro_f1"].mean()),
            "fold_macro_f1_std": float(fold_results["baseline_macro_f1"].std(ddof=0)),
        },
        "v2": {
            **v2_score.as_dict(),
            "fold_macro_f1_mean": float(fold_results["v2_macro_f1"].mean()),
            "fold_macro_f1_std": float(fold_results["v2_macro_f1"].std(ddof=0)),
            "fold_macro_f1_min": float(fold_results["v2_macro_f1"].min()),
            "mean_fold_delta": float(fold_results["macro_f1_delta"].mean()),
        },
    }
    return {
        "metrics": metrics,
        "fold_results": fold_results,
        "predictions": predictions,
        "split_manifest": pd.DataFrame(split_rows),
    }


def train_and_save_v2(
    train_dir: str | Path,
    labels_csv: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    cache_dir: str | Path,
    config: CorrugationConfig | None = None,
) -> dict[str, Any]:
    config = config or CorrugationConfig()
    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    manifest = load_training_manifest(train_dir, labels_csv)
    features = load_or_extract_training_features_v2(train_dir, manifest, cache_dir, config)
    labels = manifest["label"].to_numpy()
    groups = duplicate_groups(features)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = nested_compare(features, labels, config)
    final_spec, final_candidates = select_v2_spec(features, labels, groups, config)
    final_model = FittedCorrugationV2(final_spec, config.random_state).fit(features, labels)

    baseline = comparison["metrics"]["baseline"]
    v2 = comparison["metrics"]["v2"]
    gates = {
        "mean_macro_f1_delta": bool(v2["mean_fold_delta"] >= 0.02),
        "side_i_f1": bool(v2["per_class_f1"][SIDE_I_LABEL] >= 0.55),
        "side_i_recall": bool(v2["per_class_recall"][SIDE_I_LABEL] >= 0.60),
        "normal_f1": bool(v2["per_class_f1"][NORMAL_LABEL] >= 0.94),
        "side_ii_guardrail": bool(
            v2["per_class_f1"][SIDE_II_LABEL]
            >= baseline["per_class_f1"][SIDE_II_LABEL] - 0.03
        ),
    }
    metadata: dict[str, Any] = {
        "subsystem": "Rail Corrugation",
        "model_name": (
            "speed_balanced_side_detector_v2"
            if final_spec.feature_family == "time"
            else "spatial_side_detector_v2"
        ),
        "feature_extractor": (
            "base_v1" if final_spec.feature_family == "time" else "spatial_v2"
        ),
        "training_files": len(manifest),
        "class_counts": manifest["label"].value_counts().to_dict(),
        "test_data_used": False,
        "labels_sha256": sha256_file(labels_csv),
        "config": asdict(config),
        "feature_cache_signature": feature_cache_signature_v2(config),
        "final_spec": asdict(final_spec),
        "final_side_feature_names": final_model.feature_bases_,
        "candidate_metrics": comparison["metrics"],
        "acceptance_gates_before_robustness": {**gates, "passed": all(gates.values())},
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "competition_test": "Not read or predicted during training or model selection.",
    }
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump({"model": final_model, "metadata": metadata}, artifact_path)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    (report_dir / "training_report.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    comparison["fold_results"].to_csv(report_dir / "fold_results.csv", index=False)
    comparison["predictions"].to_csv(
        report_dir / "out_of_fold_predictions.csv", index=False
    )
    comparison["split_manifest"].to_csv(report_dir / "outer_splits.csv", index=False)
    pd.DataFrame(final_candidates).to_csv(
        report_dir / "final_inner_candidate_results.csv", index=False
    )
    features.select_dtypes(include="number").describe().transpose().to_csv(
        report_dir / "feature_diagnostics.csv"
    )
    return {"artifact_path": str(artifact_path), **metadata}
