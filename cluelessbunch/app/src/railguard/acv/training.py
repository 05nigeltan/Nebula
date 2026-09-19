"""Train-only ACV feature extraction, validation, and artifact creation."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy

from railguard.acv.config import AcvConfig
from railguard.acv.features import extract_case_features
from railguard.acv.models import FixedPhysicsRanker, LinearListwiseRanker
from railguard.acv.parsing import (
    AcvDataError,
    load_acv_case,
    load_training_manifest,
    sha256_file,
)
from railguard.acv.validation import compare_models, feature_ablation, robustness_suite

FEATURE_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def train_and_save(
    train_dir: str | Path,
    labels_csv: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    config: AcvConfig | None = None,
) -> dict[str, Any]:
    """Build ACV models using labelled Train files only."""

    config = config or AcvConfig()
    train_dir = Path(train_dir)
    labels_csv = Path(labels_csv)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    manifest = load_training_manifest(train_dir, labels_csv)
    faulty_by_file = dict(zip(manifest["filename"], manifest["faulty_car"], strict=True))

    cases = {}
    feature_frames = []
    for position, filename in enumerate(manifest["filename"], start=1):
        case = load_acv_case(train_dir / filename)
        if faulty_by_file[filename] not in case.car_ids:
            raise AcvDataError(f"Faulty car {faulty_by_file[filename]} is absent from {filename}")
        cases[filename] = case
        feature_frames.append(extract_case_features(case, config))
        print(f"Extracted ACV features {position:02d}/{len(manifest)}: {filename}", flush=True)
    features = pd.concat(feature_frames, ignore_index=True)

    comparison = compare_models(features, faulty_by_file, config)
    ablations = feature_ablation(features, faulty_by_file)
    robustness = robustness_suite(cases, faulty_by_file, config)
    ordinary = features[features["file_id"].isin(comparison["ordinary_files"])]
    if comparison["learned_accepted"]:
        final_model = LinearListwiseRanker(l2=comparison["final_l2"]).fit(ordinary, faulty_by_file)
        model_details = {
            "l2": comparison["final_l2"],
            "feature_names": list(final_model.feature_names),
            "weights": final_model.weights_.tolist(),
        }
    else:
        final_model = FixedPhysicsRanker()
        model_details = {
            "feature_names": list(final_model.feature_names),
            "weights": list(final_model.weights),
        }

    rich_rows = []
    for file_id in sorted(set(features["file_id"]).difference(comparison["ordinary_files"])):
        local = features[features["file_id"].eq(file_id)]
        ranking = FixedPhysicsRanker().rank(local)
        faulty_car = faulty_by_file[file_id]
        rank = ranking.index(faulty_car) + 1
        rich_rows.append(
            {
                "file_id": file_id,
                "faulty_car": faulty_car,
                "true_rank": rank,
                "ranked_cars": "|".join(ranking),
                "note": "Single rich-schema case; diagnostic only",
            }
        )

    package_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    metadata: dict[str, Any] = {
        "subsystem": "ACV",
        "selected_model": comparison["selected_model"],
        "training_files": len(manifest),
        "ordinary_training_files": len(comparison["ordinary_files"]),
        "test_data_used": False,
        "labels_sha256": sha256_file(labels_csv),
        "source_sha256": {file_id: cases[file_id].sha256 for file_id in manifest["filename"]},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config": asdict(config),
        "model_details": model_details,
        "candidate_metrics": {
            "peer_physics_fixed": comparison["fixed_metrics"],
            "linear_listwise": comparison["learned_metrics"],
        },
        "selection_gate": {
            "linear_score_gain": comparison["score_gain"],
            "minimum_required_gain": config.learned_min_score_gain,
            "linear_accepted": comparison["learned_accepted"],
        },
        "robustness_metrics": {
            "temporal_blocks": robustness["temporal_metrics"],
            "row_dropout": robustness["dropout_metrics"],
        },
        "rich_schema_diagnostic": rich_rows,
        "package_versions": package_versions,
        "candidate_notes": {
            "competition_test": "Not read or predicted during training.",
            "rich_schema": "Only one rich-pressure case exists; it is excluded from primary selection.",
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump({"model": final_model, "metadata": metadata}, artifact_path)
    json_text = json.dumps(metadata, indent=2, default=_json_default)
    (artifact_dir / "metadata.json").write_text(json_text, encoding="utf-8")
    (report_dir / "training_report.json").write_text(json_text, encoding="utf-8")
    features.to_csv(report_dir / "car_features.csv", index=False)
    pd.DataFrame(comparison["fixed_rows"] + comparison["learned_rows"]).to_csv(
        report_dir / "out_of_fold_rankings.csv", index=False
    )
    pd.DataFrame(comparison["inner_candidate_rows"]).to_csv(
        report_dir / "fold_results.csv", index=False
    )
    pd.DataFrame(comparison["final_candidate_rows"]).to_csv(
        report_dir / "candidate_results.csv", index=False
    )
    pd.DataFrame(ablations).to_csv(report_dir / "ablation_results.csv", index=False)
    (report_dir / "robustness_report.json").write_text(
        json.dumps(robustness, indent=2, default=_json_default), encoding="utf-8"
    )
    return {"artifact_path": str(artifact_path), **metadata}
