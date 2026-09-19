"""Feature caching, nested grouped validation, model selection, and artifact creation."""

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from railguard.corrugation.config import (
    SIDE_I_LABEL,
    SIDE_II_LABEL,
    CorrugationConfig,
)
from railguard.corrugation.features import extract_features
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import (
    FittedCorrugationModel,
    ModelSpec,
    available_side_bases,
    choose_threshold,
    feature_bases,
)
from railguard.corrugation.parsing import (
    load_corrugation_file,
    load_training_manifest,
    sha256_file,
)

FEATURE_SCHEMA_VERSION = 1


def feature_cache_signature(config: CorrugationConfig) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "expected_samples": config.expected_samples,
        "sample_rate_hz": config.sample_rate_hz,
        "tach_teeth": config.tach_teeth,
        "wheel_diameter_m": config.wheel_diameter_m,
        "welch_nperseg": config.welch_nperseg,
        "welch_overlap": config.welch_overlap,
        "stft_nperseg": config.stft_nperseg,
        "stft_overlap": config.stft_overlap,
        "fixed_bands_hz": [list(band) for band in config.fixed_bands_hz],
        "wavelength_bands_m": [list(band) for band in config.wavelength_bands_m],
    }


def _extract_training_features(
    train_dir: Path,
    manifest: pd.DataFrame,
    config: CorrugationConfig,
) -> pd.DataFrame:
    rows = []
    for position, filename in enumerate(manifest["filename"], start=1):
        signal = load_corrugation_file(train_dir / filename, config.expected_samples)
        rows.append(extract_features(signal, config))
        print(
            f"Extracted corrugation features {position:03d}/{len(manifest)}: {filename}",
            flush=True,
        )
    return pd.DataFrame(rows)


def load_or_extract_training_features(
    train_dir: str | Path,
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    config: CorrugationConfig,
) -> pd.DataFrame:
    """Reuse features only if the configuration and every source hash still match."""

    train_dir = Path(train_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "train_features.csv"
    metadata_path = cache_dir / "cache_metadata.json"
    cached_signature = None
    if metadata_path.is_file():
        try:
            cached_signature = json.loads(metadata_path.read_text(encoding="utf-8"))[
                "feature_signature"
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            cached_signature = None
    if cache_path.is_file() and cached_signature == feature_cache_signature(config):
        cached = pd.read_csv(cache_path)
        ids_match = (
            cached.get("file_id", pd.Series(dtype=str)).tolist() == manifest["filename"].tolist()
        )
        if {"file_id", "sha256"}.issubset(cached.columns) and ids_match:
            current_hashes = [sha256_file(train_dir / name) for name in manifest["filename"]]
            if cached["sha256"].tolist() == current_hashes:
                print("Reusing verified corrugation feature cache", flush=True)
                return cached
    features = _extract_training_features(train_dir, manifest, config)
    features.to_csv(cache_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_signature": feature_cache_signature(config),
                "source_sha256": dict(zip(features["file_id"], features["sha256"], strict=True)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return features


def duplicate_groups(features: pd.DataFrame) -> np.ndarray:
    """Use content hashes as groups so identical recordings cannot cross folds."""

    first_id_by_hash: dict[str, str] = {}
    groups = []
    for file_id, digest in features[["file_id", "sha256"]].itertuples(index=False):
        groups.append(first_id_by_hash.setdefault(digest, file_id))
    return np.asarray(groups)


def _candidate_specs(config: CorrugationConfig):
    for family in config.feature_families:
        for c_value in config.c_values:
            for multiplier in config.positive_weight_multipliers:
                for include_speed in config.include_raw_speed_options:
                    yield ModelSpec(family, c_value, multiplier, include_speed)


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


def select_model_spec(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: CorrugationConfig,
) -> tuple[ModelSpec, list[dict[str, Any]]]:
    """Choose all model decisions from grouped cross-fitted training scores."""

    candidate_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_spec: ModelSpec | None = None
    complexity = {"time": 0, "compact": 1, "all": 2}
    for raw_spec in _candidate_specs(config):
        first_scores = np.empty(len(labels), dtype=float)
        second_scores = np.empty(len(labels), dtype=float)
        for train, validation in _inner_splits(labels, groups, config):
            model = FittedCorrugationModel(raw_spec, config.random_state).fit(
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
        selected_spec = replace(raw_spec, threshold=threshold, side_i_bias=side_i_bias)
        feature_count = len(feature_bases(features, raw_spec.feature_family))
        row = {
            **asdict(selected_spec),
            "inner_macro_f1": score.macro_f1,
            "inner_side_i_recall": score.per_class_recall[SIDE_I_LABEL],
            "inner_side_ii_recall": score.per_class_recall[SIDE_II_LABEL],
            "side_feature_count": feature_count,
        }
        candidate_rows.append(row)
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
    candidate_rows.sort(key=lambda row: row["inner_macro_f1"], reverse=True)
    return best_spec, candidate_rows


def _direct_matrix(features: pd.DataFrame) -> np.ndarray:
    bases = available_side_bases(features)
    first = features[[f"side_i_{base}" for base in bases]].to_numpy(float)
    second = features[[f"side_ii_{base}" for base in bases]].to_numpy(float)
    return np.column_stack((first, second, first - second, np.abs(first - second)))


def _baseline_predictions(
    features: pd.DataFrame,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    direct_x = _direct_matrix(features)
    direct = make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.03, class_weight="balanced", dual="auto", max_iter=20_000),
    ).fit(direct_x[train], labels[train])
    speed_x = features[["tach_transitions", "speed_mps", "tach_duty"]].to_numpy(float)
    speed = make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.03, class_weight="balanced", dual="auto", max_iter=20_000),
    ).fit(speed_x[train], labels[train])
    return direct.predict(direct_x[validation]), speed.predict(speed_x[validation])


def nested_validate(
    features: pd.DataFrame,
    labels: np.ndarray,
    config: CorrugationConfig,
) -> dict[str, Any]:
    groups = duplicate_groups(features)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            spec, _ = select_model_spec(
                features.iloc[train].reset_index(drop=True),
                labels[train],
                groups[train],
                config,
            )
            model = FittedCorrugationModel(spec, config.random_state + repeat).fit(
                features.iloc[train], labels[train]
            )
            first, second = model.decision_scores(features.iloc[validation])
            paired_prediction = model.predict(features.iloc[validation])
            direct_prediction, speed_prediction = _baseline_predictions(
                features, labels, train, validation
            )
            paired_score = score_corrugation(labels[validation], paired_prediction)
            fold_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    **asdict(spec),
                    **paired_score.as_dict(),
                }
            )
            for local, file_index in enumerate(validation):
                prediction_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": features.iloc[file_index]["file_id"],
                        "truth": labels[file_index],
                        "side_i_score": first[local],
                        "side_ii_score": second[local],
                        "paired_prediction": paired_prediction[local],
                        "direct_prediction": direct_prediction[local],
                        "speed_prediction": speed_prediction[local],
                    }
                )
        print(
            f"Completed nested corrugation validation repeat {repeat + 1}/{config.outer_repeats}",
            flush=True,
        )
    predictions = pd.DataFrame(prediction_rows)
    fold_results = pd.DataFrame(fold_rows)
    candidate_scores = {
        name: score_corrugation(
            predictions["truth"].to_numpy(), predictions[column].to_numpy()
        ).as_dict()
        for name, column in {
            "speed_only": "speed_prediction",
            "direct_three_class_linear_svm": "direct_prediction",
            "side_symmetric_linear_svm": "paired_prediction",
        }.items()
    }
    candidate_scores["side_symmetric_linear_svm"].update(
        {
            "fold_macro_f1_mean": float(fold_results["macro_f1"].mean()),
            "fold_macro_f1_std": float(fold_results["macro_f1"].std(ddof=0)),
            "fold_macro_f1_min": float(fold_results["macro_f1"].min()),
        }
    )
    return {
        "candidate_scores": candidate_scores,
        "fold_results": fold_results,
        "predictions": predictions,
    }


def train_and_save(
    train_dir: str | Path,
    labels_csv: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    cache_dir: str | Path,
    config: CorrugationConfig | None = None,
) -> dict[str, Any]:
    """Train only on labelled recordings and save a reproducible artifact."""

    config = config or CorrugationConfig()
    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    manifest = load_training_manifest(train_dir, labels_csv)
    features = load_or_extract_training_features(train_dir, manifest, cache_dir, config)
    labels = manifest["label"].to_numpy()
    groups = duplicate_groups(features)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    validation = nested_validate(features, labels, config)
    final_spec, final_candidates = select_model_spec(features, labels, groups, config)
    final_model = FittedCorrugationModel(final_spec, config.random_state).fit(features, labels)

    paired_score = validation["candidate_scores"]["side_symmetric_linear_svm"]
    acceptance = {
        "mean_macro_f1": bool(paired_score["fold_macro_f1_mean"] >= config.minimum_mean_macro_f1),
        "side_i_recall": bool(
            paired_score["per_class_recall"][SIDE_I_LABEL] >= config.minimum_side_i_recall
        ),
        "side_ii_recall": bool(
            paired_score["per_class_recall"][SIDE_II_LABEL] >= config.minimum_side_ii_recall
        ),
    }
    metadata: dict[str, Any] = {
        "subsystem": "Rail Corrugation",
        "model_name": "side_symmetric_linear_svm",
        "training_files": len(manifest),
        "class_counts": manifest["label"].value_counts().to_dict(),
        "test_data_used": False,
        "labels_sha256": sha256_file(labels_csv),
        "duplicate_groups": [
            group["file_id"].tolist()
            for _, group in features.groupby("sha256", sort=False)
            if len(group) > 1
        ],
        "config": asdict(config),
        "feature_cache_signature": feature_cache_signature(config),
        "final_spec": asdict(final_spec),
        "final_side_feature_names": final_model.feature_bases_,
        "candidate_metrics": validation["candidate_scores"],
        "acceptance_gates_before_overlap_test": {
            **acceptance,
            "passed": all(acceptance.values()),
        },
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
    (report_dir / "training_report.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    validation["fold_results"].to_csv(report_dir / "fold_results.csv", index=False)
    validation["predictions"].to_csv(report_dir / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(
        [{"candidate": name, **values} for name, values in validation["candidate_scores"].items()]
    ).to_json(report_dir / "candidate_results.json", orient="records", indent=2)
    pd.DataFrame(final_candidates).to_csv(
        report_dir / "final_inner_candidate_results.csv", index=False
    )
    features.select_dtypes(include="number").describe().transpose().to_csv(
        report_dir / "feature_diagnostics.csv"
    )
    return {"artifact_path": str(artifact_path), **metadata}
