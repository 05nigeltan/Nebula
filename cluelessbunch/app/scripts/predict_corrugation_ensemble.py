"""Fit the train-selected ensemble and export a separate, backed-up submission."""

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.ensemble_experiment import fit_pair, select_ensemble
from railguard.corrugation.inference import predict_corrugation_directory, write_predictions_atomic
from railguard.corrugation.models_ensemble import FittedCorrugationEnsemble
from railguard.corrugation.parsing import load_training_manifest, sha256_file
from railguard.corrugation.training import duplicate_groups, load_or_extract_training_features
from railguard.corrugation.training_augmented import (
    _view_training_arrays,
    load_or_extract_training_views,
)
from railguard.project_paths import participant_root


def main():
    root = Path(__file__).resolve().parents[1]
    prediction_dir = root.parent / "Optional_Items/predictions"
    previous = prediction_dir / "rail_predictions.csv"
    output = prediction_dir / "rail_predictions_ensemble.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; archive or rename it first")
    previous_hash = sha256_file(previous)
    previous_frame = pd.read_csv(previous)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run = root / "reports/corrugation_ensemble/submissions" / stamp
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(previous, run / "rail_predictions_previous.csv")
    config = CorrugationConfig()
    data = participant_root(root) / "02_Datasets/Rail_Corrugation"
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    frame = load_or_extract_training_features(
        data / "Train", manifest, root / "cache/corrugation", config
    )
    full, views = load_or_extract_training_views(
        data / "Train", manifest, root / "cache/corrugation_aug", config
    )
    if (
        frame.file_id.tolist() != full.file_id.tolist()
        or frame.sha256.tolist() != full.sha256.tolist()
    ):
        raise ValueError("Feature source caches disagree")
    labels = manifest.label.to_numpy()
    spec, weight = select_ensemble(frame, views, labels, duplicate_groups(frame), config)[
        "sensor_ensemble"
    ]
    print(f"Training-only selected blend: {weight}, spec: {spec}", flush=True)
    arrays = _view_training_arrays(
        views, frame.file_id.to_numpy(), dict(zip(frame.file_id, labels, strict=True))
    )
    base, augmented, alignment = fit_pair(frame, labels, arrays, spec, config.random_state)
    model = FittedCorrugationEnsemble(base, augmented, alignment, weight)
    metadata = {
        "subsystem": "Rail Corrugation",
        "config": asdict(config),
        "final_spec": asdict(spec),
        "augmented_weight": weight,
        "alignment": alignment,
        "training_files": len(frame),
        "test_used_for_selection": False,
        "experimental_not_deployed": True,
        "previous_sha256": previous_hash,
        "labels_sha256": sha256_file(data / "Train_Labels.csv"),
    }
    artifact = run / "model.joblib"
    joblib.dump({"model": model, "metadata": metadata}, artifact)
    predictions, diagnostics = predict_corrugation_directory(data / "Test", artifact)
    if set(predictions.file_id) != set(previous_frame.file_id):
        raise ValueError("New and previous predictions cover different files")
    merged = previous_frame.merge(
        predictions, on="file_id", suffixes=("_previous", "_ensemble"), validate="one_to_one"
    )
    changed = merged[merged.prediction_previous != merged.prediction_ensemble]
    assert sha256_file(previous) == previous_hash
    assert sha256_file(run / "rail_predictions_previous.csv") == previous_hash
    write_predictions_atomic(predictions, output)
    write_predictions_atomic(predictions, run / "rail_predictions.csv")
    changed.to_csv(run / "changed_predictions.csv", index=False)
    diagnostics.to_csv(run / "diagnostics.csv", index=False)
    metadata.update(
        {
            "output": str(output),
            "run_directory": str(run),
            "prediction_rows": len(predictions),
            "changed_predictions": len(changed),
            "class_counts": predictions.prediction.value_counts().to_dict(),
            "output_sha256": sha256_file(output),
        }
    )
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
