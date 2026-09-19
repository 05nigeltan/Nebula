"""Promote the user-approved ensemble, preserving every replaced artifact."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/corrugation_ensemble/submissions/20260919T041338318951Z"
    bundle = root.parent
    prediction_dir = bundle / "Optional_Items/predictions"
    backup = (
        root
        / "reports/corrugation_ensemble/promotions"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    artifact = joblib.load(source / "model.joblib")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "experimental_not_deployed": False,
            "model_name": "side_symmetric_linear_svm_sensor_view_ensemble",
            "promotion_basis": "User approved after reporting submission macro-F1 0.68 versus previous 0.60",
            "submission_scores_user_reported": {"ensemble": 0.68, "previous": 0.60},
            "test_used_for_selection": True,
            "selection_note": "Hyperparameters fit using training data only; final model choice uses reported submission feedback.",
            "local_mean_fold_macro_f1": 0.7543337139834344,
            "local_control_mean_fold_macro_f1": 0.7554361670009376,
        }
    )
    artifact["metadata"] = metadata
    destinations = [
        root / "artifacts/corrugation",
        bundle / "Optional_Items/Rail Corrugation/model",
    ]
    files = [p / n for p in destinations for n in ("model.joblib", "metadata.json")]
    files += [
        prediction_dir / "rail_predictions.csv",
        prediction_dir / "predictions.zip",
        bundle / "Optional_Items/packaging_manifest.json",
    ]
    for path in files:
        if path.exists():
            dest = backup / path.relative_to(bundle)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    for dest in destinations:
        joblib.dump(artifact, dest / "model.joblib")
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(
        prediction_dir / "rail_predictions_ensemble.csv", prediction_dir / "rail_predictions.csv"
    )
    shutil.copy2(bundle / "predictions.zip", prediction_dir / "predictions.zip")
    manifest_path = bundle / "Optional_Items/packaging_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["app_rail_model"] = (
        "Promoted 75% baseline / 25% sensor-view ensemble; matches prediction ZIP"
    )
    manifest["promotion"] = metadata["promotion_basis"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(root / "RAILGUARD_TECHNICAL_WRITEUP.md", bundle / "Optional_Items/write_up.md")
    shutil.copy2(
        root / "CORRUGATION_README.md",
        bundle / "Optional_Items/documentation/CORRUGATION_README.md",
    )
    print(f"Promoted ensemble. Rollback files: {backup}")


if __name__ == "__main__":
    main()
