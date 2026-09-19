"""Train-only SHM feature extraction, nested validation, and artifact creation."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rainflow
import sklearn

from railguard.shm.config import ShmConfig
from railguard.shm.models import FittedDamageModel
from railguard.shm.parsing import load_shm_file, load_training_manifest, sha256_file
from railguard.shm.rainflow_features import extract_features
from railguard.shm.validation import nested_validate

FEATURE_SCHEMA_VERSION = 1


def feature_cache_signature(config: ShmConfig) -> dict[str, Any]:
    """Return only settings that can change extracted feature values."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "expected_samples": config.expected_samples,
        "exponents": list(config.exponents),
        "amplitude_bins": list(config.amplitude_bins),
    }


def _extract_training_features(
    train_dir: Path,
    manifest: pd.DataFrame,
    config: ShmConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        signal = load_shm_file(train_dir / row.filename, config.expected_samples)
        rows.append(extract_features(signal, config))
        print(f"Extracted SHM features {position:02d}/{len(manifest)}: {row.filename}", flush=True)
    return pd.DataFrame(rows)


def _load_or_extract_training_features(
    train_dir: Path,
    manifest: pd.DataFrame,
    cache_dir: Path,
    config: ShmConfig,
) -> pd.DataFrame:
    """Reuse features only when every cached source hash still matches."""

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
        required = {"file_id", "sha256", *(f"log_b_m_{value:g}" for value in config.exponents)}
        file_ids_match = (
            cached.get("file_id", pd.Series(dtype=str)).tolist() == manifest["filename"].tolist()
        )
        if required.issubset(cached.columns) and file_ids_match:
            current_hashes = [sha256_file(train_dir / name) for name in manifest["filename"]]
            if cached["sha256"].tolist() == current_hashes:
                print("Reusing verified SHM training feature cache", flush=True)
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


def train_and_save(
    train_dir: str | Path,
    labels_csv: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    cache_dir: str | Path,
    config: ShmConfig | None = None,
) -> dict[str, Any]:
    """Train only on labelled files; the competition Test directory is never accepted here."""

    config = config or ShmConfig()
    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    cache_dir = Path(cache_dir)
    manifest = load_training_manifest(train_dir, labels_csv)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features = _load_or_extract_training_features(train_dir, manifest, cache_dir, config)
    if features["file_id"].tolist() != manifest["filename"].tolist():
        raise RuntimeError("Extracted feature order does not match the label manifest")
    target = manifest["damage"].to_numpy(float)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    validation = nested_validate(features, target, config)
    final_model = FittedDamageModel(
        validation.final_physics_spec,
        validation.final_residual_spec,
        config,
    ).fit(features, target)

    candidate_metrics = {
        "median_baseline": validation.median_metrics.as_dict(),
        "rainflow_physics": validation.physics_metrics.as_dict(),
        "gated_hybrid_candidate": validation.hybrid_metrics.as_dict(),
        "selected_model": validation.selected_metrics.as_dict(),
    }
    selected_name = "hybrid" if validation.hybrid_accepted else "rainflow_physics"
    package_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "rainflow": getattr(rainflow, "__version__", "unknown"),
    }
    final_parameters = {
        "physics_intercept": final_model.physics_.intercept_,
        "physics_slope": final_model.physics_.slope_,
        "physics_scale": float(np.exp(final_model.physics_.intercept_)),
        "residual_feature_names": final_model.feature_names_,
    }
    target_reference = {
        "minimum": float(np.min(target)),
        "q25": float(np.quantile(target, 0.25)),
        "median": float(np.median(target)),
        "q75": float(np.quantile(target, 0.75)),
        "q90": float(np.quantile(target, 0.90)),
        "maximum": float(np.max(target)),
        "meaning": "Distribution of labelled training damage targets; not a safety threshold.",
    }
    metadata: dict[str, Any] = {
        "subsystem": "SHM",
        "selected_model": selected_name,
        "training_files": len(manifest),
        "test_data_used": False,
        "labels_sha256": sha256_file(labels_csv),
        "config": asdict(config),
        "feature_columns": [
            column for column in features.columns if column not in {"file_id", "sha256"}
        ],
        "feature_cache_signature": feature_cache_signature(config),
        "final_physics_spec": asdict(validation.final_physics_spec),
        "final_residual_spec": asdict(validation.final_residual_spec),
        "final_parameters": final_parameters,
        "target_reference": target_reference,
        "hybrid_gate": validation.gate,
        "candidate_metrics": candidate_metrics,
        "package_versions": package_versions,
        "candidate_notes": {
            "huber": "Not eligible in this run after repeated optimizer non-convergence.",
            "competition_test": "Not read or predicted during training.",
        },
    }
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump({"model": final_model, "metadata": metadata}, artifact_path)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    (report_dir / "training_report.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )

    candidate_rows = []
    for name, metrics in candidate_metrics.items():
        candidate_rows.append({"candidate": name, **metrics})
    pd.DataFrame(candidate_rows).to_csv(report_dir / "candidate_results.csv", index=False)
    pd.DataFrame(validation.fold_results).to_csv(report_dir / "fold_results.csv", index=False)
    prediction_frame = pd.DataFrame(
        {
            "file_id": manifest["filename"],
            "truth": target,
            "median_prediction": validation.median_predictions,
            "physics_prediction": validation.physics_predictions,
            "hybrid_prediction": validation.hybrid_predictions,
            "selected_prediction": validation.selected_predictions,
        }
    )
    prediction_frame["selected_ape"] = (
        np.abs(prediction_frame["selected_prediction"] - prediction_frame["truth"])
        / prediction_frame["truth"]
    )
    prediction_frame.to_csv(report_dir / "out_of_fold_predictions.csv", index=False)
    features.select_dtypes(include="number").describe().transpose().to_csv(
        report_dir / "feature_diagnostics.csv"
    )
    return {"artifact_path": str(artifact_path), **metadata}
