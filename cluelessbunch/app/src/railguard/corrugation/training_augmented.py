"""Fold-safe sensor-view augmentation and locked comparison against corrugation v1."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.augmentation import (
    extract_jackknife_views,
    full_features_from_views,
    select_views,
    validate_view_groups,
)
from railguard.corrugation.config import SIDE_I_LABEL, SIDE_II_LABEL, CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import (
    FittedCorrugationModel,
    ModelSpec,
    choose_threshold,
    feature_bases,
)
from railguard.corrugation.models_augmented import FittedAugmentedCorrugationModel
from railguard.corrugation.parsing import (
    load_corrugation_file,
    load_training_manifest,
    sha256_file,
)
from railguard.corrugation.training import (
    duplicate_groups,
    feature_cache_signature,
    select_model_spec,
)

AUGMENTATION_SCHEMA_VERSION = 1


def augmentation_cache_signature(config: CorrugationConfig) -> dict[str, Any]:
    return {
        "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
        "view_policy": "full_plus_leave_one_car_out",
        "cars": list(range(1, 9)),
        "base_feature_signature": feature_cache_signature(config),
    }


def _extract_training_views(
    train_dir: Path,
    manifest: pd.DataFrame,
    config: CorrugationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, filename in enumerate(manifest["filename"], start=1):
        signal = load_corrugation_file(train_dir / filename, config.expected_samples)
        rows.extend(extract_jackknife_views(signal, config))
        print(
            f"Extracted corrugation sensor views {position:03d}/{len(manifest)}: {filename}",
            flush=True,
        )
    views = pd.DataFrame(rows)
    validate_view_groups(views)
    return views


def load_or_extract_training_views(
    train_dir: str | Path,
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    config: CorrugationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load verified full/view features or extract them from labelled recordings."""

    train_dir = Path(train_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_path = cache_dir / "train_full_features.csv"
    views_path = cache_dir / "train_view_features.csv"
    metadata_path = cache_dir / "cache_metadata.json"
    signature = augmentation_cache_signature(config)

    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    if (
        full_path.is_file()
        and views_path.is_file()
        and metadata.get("feature_signature") == signature
    ):
        full = pd.read_csv(full_path)
        views = pd.read_csv(views_path)
        ids_match = full.get("file_id", pd.Series(dtype=str)).tolist() == manifest[
            "filename"
        ].tolist()
        current_hashes = [sha256_file(train_dir / name) for name in manifest["filename"]]
        if (
            ids_match
            and {"file_id", "sha256"}.issubset(full.columns)
            and full["sha256"].tolist() == current_hashes
        ):
            validate_view_groups(views)
            print("Reusing verified corrugation augmentation feature cache", flush=True)
            return full, views

    views = _extract_training_views(train_dir, manifest, config)
    full = full_features_from_views(views)
    full.to_csv(full_path, index=False)
    views.to_csv(views_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_signature": signature,
                "source_sha256": dict(zip(full["file_id"], full["sha256"], strict=True)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return full, views


def _candidate_specs(config: CorrugationConfig):
    for family in config.feature_families:
        for c_value in config.c_values:
            for multiplier in config.positive_weight_multipliers:
                for include_speed in config.include_raw_speed_options:
                    yield ModelSpec(family, c_value, multiplier, include_speed)


def _view_training_arrays(
    views: pd.DataFrame,
    source_ids: np.ndarray,
    label_by_file: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selected = select_views(views, source_ids)
    labels = selected["source_file_id"].map(label_by_file)
    if labels.isna().any():
        raise ValueError("A sensor view has no source-file label")
    return (
        selected,
        labels.to_numpy(),
        selected["view_weight"].to_numpy(dtype=float),
    )


def select_augmented_spec(
    full_features: pd.DataFrame,
    views: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: CorrugationConfig,
) -> tuple[ModelSpec, list[dict[str, Any]]]:
    """Select model and threshold with augmented inner training and untouched validation."""

    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    label_by_file = dict(zip(full_features["file_id"], labels, strict=True))
    prepared = []
    for train, validation in splitter.split(np.zeros(len(labels)), labels, groups):
        source_ids = full_features.iloc[train]["file_id"].to_numpy(str)
        train_views, train_labels, train_weights = _view_training_arrays(
            views, source_ids, label_by_file
        )
        prepared.append((train_views, train_labels, train_weights, validation))

    best_key: tuple[float, float, float, float] | None = None
    best_spec: ModelSpec | None = None
    rows: list[dict[str, Any]] = []
    complexity = {"time": 0, "compact": 1, "all": 2}
    for raw_spec in _candidate_specs(config):
        first_scores = np.empty(len(labels), dtype=float)
        second_scores = np.empty(len(labels), dtype=float)
        for train_views, train_labels, train_weights, validation in prepared:
            model = FittedAugmentedCorrugationModel(raw_spec, config.random_state).fit(
                train_views, train_labels, train_weights
            )
            first, second = model.decision_scores(full_features.iloc[validation])
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
        selected_spec = replace(raw_spec, threshold=threshold, side_i_bias=side_i_bias)
        rows.append(
            {
                **asdict(selected_spec),
                "inner_macro_f1": score.macro_f1,
                "inner_side_i_recall": score.per_class_recall[SIDE_I_LABEL],
                "inner_side_ii_recall": score.per_class_recall[SIDE_II_LABEL],
                "side_feature_count": len(
                    feature_bases(full_features, raw_spec.feature_family)
                ),
            }
        )
        minimum_fault_recall = min(
            score.per_class_recall[SIDE_I_LABEL], score.per_class_recall[SIDE_II_LABEL]
        )
        key = (
            score.macro_f1,
            minimum_fault_recall,
            float(not raw_spec.include_raw_speed),
            -float(complexity[raw_spec.feature_family]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_spec = selected_spec
    assert best_spec is not None
    rows.sort(key=lambda row: row["inner_macro_f1"], reverse=True)
    return best_spec, rows


def _metric_summary(
    predictions: pd.DataFrame,
    prediction_column: str,
    fold_results: pd.DataFrame,
    fold_column: str,
) -> dict[str, Any]:
    result = score_corrugation(
        predictions["truth"].to_numpy(), predictions[prediction_column].to_numpy()
    ).as_dict()
    result.update(
        {
            "fold_macro_f1_mean": float(fold_results[fold_column].mean()),
            "fold_macro_f1_std": float(fold_results[fold_column].std(ddof=0)),
            "fold_macro_f1_min": float(fold_results[fold_column].min()),
        }
    )
    return result


def nested_compare_augmented(
    full_features: pd.DataFrame,
    views: pd.DataFrame,
    labels: np.ndarray,
    config: CorrugationConfig,
) -> dict[str, Any]:
    """Compare v1 and augmented training on identical grouped outer folds."""

    groups = duplicate_groups(full_features)
    label_by_file = dict(zip(full_features["file_id"], labels, strict=True))
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    dropout_rows: list[dict[str, Any]] = []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            train_full = full_features.iloc[train].reset_index(drop=True)
            train_labels = labels[train]
            train_groups = groups[train]
            baseline_spec, _ = select_model_spec(
                train_full, train_labels, train_groups, config
            )
            augmented_spec, _ = select_augmented_spec(
                train_full, views, train_labels, train_groups, config
            )
            baseline = FittedCorrugationModel(
                baseline_spec, config.random_state + repeat
            ).fit(train_full, train_labels)
            source_ids = train_full["file_id"].to_numpy(str)
            train_views, view_labels, view_weights = _view_training_arrays(
                views, source_ids, label_by_file
            )
            augmented = FittedAugmentedCorrugationModel(
                augmented_spec, config.random_state + repeat
            ).fit(train_views, view_labels, view_weights)

            validation_full = full_features.iloc[validation]
            baseline_prediction = baseline.predict(validation_full)
            augmented_prediction = augmented.predict(validation_full)
            first, second = augmented.decision_scores(validation_full)
            baseline_score = score_corrugation(labels[validation], baseline_prediction)
            augmented_score = score_corrugation(labels[validation], augmented_prediction)
            fold_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "baseline_macro_f1": baseline_score.macro_f1,
                    "augmented_macro_f1": augmented_score.macro_f1,
                    "delta": augmented_score.macro_f1 - baseline_score.macro_f1,
                    "baseline_spec": json.dumps(asdict(baseline_spec)),
                    "augmented_spec": json.dumps(asdict(augmented_spec)),
                }
            )
            for local, file_index in enumerate(validation):
                prediction_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": full_features.iloc[file_index]["file_id"],
                        "truth": labels[file_index],
                        "speed_mps": full_features.iloc[file_index]["speed_mps"],
                        "side_i_score": first[local],
                        "side_ii_score": second[local],
                        "baseline_prediction": baseline_prediction[local],
                        "augmented_prediction": augmented_prediction[local],
                    }
                )

            validation_ids = validation_full["file_id"].to_numpy(str)
            validation_views = views.loc[
                views["source_file_id"].isin(validation_ids)
                & (views["view_id"] != "full")
            ].copy()
            validation_views["truth"] = validation_views["source_file_id"].map(label_by_file)
            validation_views["prediction"] = augmented.predict(validation_views)
            for row in validation_views[
                ["source_file_id", "omitted_car", "truth", "prediction"]
            ].itertuples(index=False):
                dropout_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": row.source_file_id,
                        "omitted_car": int(row.omitted_car),
                        "truth": row.truth,
                        "prediction": row.prediction,
                    }
                )
        print(
            f"Completed augmented corrugation validation repeat "
            f"{repeat + 1}/{config.outer_repeats}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    dropouts = pd.DataFrame(dropout_rows)
    baseline_metrics = _metric_summary(
        predictions, "baseline_prediction", folds, "baseline_macro_f1"
    )
    augmented_metrics = _metric_summary(
        predictions, "augmented_prediction", folds, "augmented_macro_f1"
    )
    augmented_metrics["mean_fold_delta"] = float(folds["delta"].mean())
    return {
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "fold_results": folds,
        "predictions": predictions,
        "dropout_predictions": dropouts,
    }


def paired_file_bootstrap(
    predictions: pd.DataFrame,
    iterations: int = 2_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap files within repeat and compare paired macro-F1 values."""

    rng = np.random.default_rng(random_state)
    deltas = []
    repeats = predictions["repeat"].nunique()
    for _, repeat_frame in predictions.groupby("repeat", sort=True):
        repeat_frame = repeat_frame.reset_index(drop=True)
        for _ in range(iterations // repeats):
            sampled = rng.integers(0, len(repeat_frame), len(repeat_frame))
            sample = repeat_frame.iloc[sampled]
            baseline = score_corrugation(
                sample["truth"].to_numpy(), sample["baseline_prediction"].to_numpy()
            ).macro_f1
            augmented = score_corrugation(
                sample["truth"].to_numpy(), sample["augmented_prediction"].to_numpy()
            ).macro_f1
            deltas.append(augmented - baseline)
    values = np.asarray(deltas)
    return {
        "iterations": len(values),
        "median_delta": float(np.median(values)),
        "p05_delta": float(np.quantile(values, 0.05)),
        "p95_delta": float(np.quantile(values, 0.95)),
        "probability_augmented_better": float(np.mean(values > 0)),
    }


def _dropout_summary(dropouts: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for omitted_car, frame in dropouts.groupby("omitted_car", sort=True):
        rows.append(
            {
                "omitted_car": int(omitted_car),
                **score_corrugation(
                    frame["truth"].to_numpy(), frame["prediction"].to_numpy()
                ).as_dict(),
            }
        )
    return rows


def _tune_fixed_augmented_spec(
    full_features: pd.DataFrame,
    views: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    spec: ModelSpec,
    config: CorrugationConfig,
) -> ModelSpec:
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    raw_spec = replace(spec, threshold=0.0, side_i_bias=0.0)
    label_by_file = dict(zip(full_features["file_id"], labels, strict=True))
    first_scores = np.empty(len(labels), dtype=float)
    second_scores = np.empty(len(labels), dtype=float)
    for train, validation in splitter.split(np.zeros(len(labels)), labels, groups):
        source_ids = full_features.iloc[train]["file_id"].to_numpy(str)
        train_views, train_labels, train_weights = _view_training_arrays(
            views, source_ids, label_by_file
        )
        model = FittedAugmentedCorrugationModel(raw_spec, config.random_state).fit(
            train_views, train_labels, train_weights
        )
        first, second = model.decision_scores(full_features.iloc[validation])
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


def speed_bin_holdouts(
    full_features: pd.DataFrame,
    views: pd.DataFrame,
    labels: np.ndarray,
    spec: ModelSpec,
    config: CorrugationConfig,
) -> list[dict[str, Any]]:
    speed = full_features["speed_mps"].to_numpy(float)
    groups = duplicate_groups(full_features)
    label_by_file = dict(zip(full_features["file_id"], labels, strict=True))
    rows = []
    for lower, upper in ((9.5, 12.0), (12.0, 15.0), (15.0, np.inf)):
        validation = np.flatnonzero((speed >= lower) & (speed < upper))
        train = np.flatnonzero(~((speed >= lower) & (speed < upper)))
        train_full = full_features.iloc[train].reset_index(drop=True)
        tuned = _tune_fixed_augmented_spec(
            train_full, views, labels[train], groups[train], spec, config
        )
        train_views, train_labels, train_weights = _view_training_arrays(
            views, train_full["file_id"].to_numpy(str), label_by_file
        )
        model = FittedAugmentedCorrugationModel(tuned, config.random_state).fit(
            train_views, train_labels, train_weights
        )
        prediction = model.predict(full_features.iloc[validation])
        rows.append(
            {
                "lower_mps": lower,
                "upper_mps": upper,
                "files": len(validation),
                "class_counts": pd.Series(labels[validation]).value_counts().to_dict(),
                **score_corrugation(labels[validation], prediction).as_dict(),
            }
        )
    return rows


def train_and_save_augmented(
    train_dir: str | Path,
    labels_csv: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    cache_dir: str | Path,
    config: CorrugationConfig | None = None,
) -> dict[str, Any]:
    """Train and evaluate the alternative without accessing competition test files."""

    config = config or CorrugationConfig()
    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    manifest = load_training_manifest(train_dir, labels_csv)
    full, views = load_or_extract_training_views(train_dir, manifest, cache_dir, config)
    labels = manifest["label"].to_numpy()
    groups = duplicate_groups(full)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = nested_compare_augmented(full, views, labels, config)
    final_spec, final_candidates = select_augmented_spec(full, views, labels, groups, config)
    label_by_file = dict(zip(full["file_id"], labels, strict=True))
    final_views, final_labels, final_weights = _view_training_arrays(
        views, full["file_id"].to_numpy(str), label_by_file
    )
    final_model = FittedAugmentedCorrugationModel(final_spec, config.random_state).fit(
        final_views, final_labels, final_weights
    )

    bootstrap = paired_file_bootstrap(comparison["predictions"], random_state=config.random_state)
    dropout = _dropout_summary(comparison["dropout_predictions"])
    speed_holdout = speed_bin_holdouts(full, views, labels, final_spec, config)
    baseline = comparison["baseline"]
    augmented = comparison["augmented"]
    worst_dropout = min(row["macro_f1"] for row in dropout)
    high_speed = speed_holdout[-1]
    gates = {
        "mean_macro_f1_delta": bool(augmented["mean_fold_delta"] >= 0.02),
        "side_i_f1": bool(augmented["per_class_f1"][SIDE_I_LABEL] >= 0.55),
        "side_ii_guardrail": bool(
            augmented["per_class_f1"][SIDE_II_LABEL]
            >= baseline["per_class_f1"][SIDE_II_LABEL] - 0.03
        ),
        "bootstrap_median_delta": bool(bootstrap["median_delta"] > 0),
        "high_speed_side_i_recall": bool(
            high_speed["per_class_recall"][SIDE_I_LABEL] > 0
        ),
        "sensor_dropout_guardrail": bool(
            worst_dropout >= augmented["fold_macro_f1_mean"] - 0.08
        ),
    }
    passed = all(gates.values())
    metadata: dict[str, Any] = {
        "subsystem": "Rail Corrugation",
        "model_name": "side_symmetric_linear_svm_sensor_jackknife",
        "training_source_files": len(manifest),
        "training_view_rows": len(views),
        "independent_training_units": len(manifest),
        "class_counts": manifest["label"].value_counts().to_dict(),
        "test_data_used": False,
        "labels_sha256": sha256_file(labels_csv),
        "config": asdict(config),
        "feature_cache_signature": augmentation_cache_signature(config),
        "final_spec": asdict(final_spec),
        "final_side_feature_names": final_model.feature_bases_,
        "candidate_metrics": {"baseline": baseline, "augmented": augmented},
        "paired_file_bootstrap": bootstrap,
        "sensor_dropout": dropout,
        "speed_bin_holdouts": speed_holdout,
        "acceptance": {**gates, "passed": passed},
        "promotion_decision": "promote_augmented" if passed else "reject_keep_v1",
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "competition_test": "Not read or predicted during training.",
    }
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump({"model": final_model, "metadata": metadata}, artifact_path)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    (report_dir / "comparison_report.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    comparison["fold_results"].to_csv(report_dir / "fold_results.csv", index=False)
    comparison["predictions"].to_csv(
        report_dir / "out_of_fold_predictions.csv", index=False
    )
    comparison["dropout_predictions"].to_csv(
        report_dir / "sensor_dropout_predictions.csv", index=False
    )
    pd.DataFrame(dropout).to_json(
        report_dir / "sensor_dropout_report.json", orient="records", indent=2
    )
    pd.DataFrame(final_candidates).to_csv(
        report_dir / "final_inner_candidate_results.csv", index=False
    )
    return {"artifact_path": str(artifact_path), **metadata}
