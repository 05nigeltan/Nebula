"""Run train-only uncertainty and speed-shift checks for corrugation v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.inference import load_corrugation_artifact
from railguard.corrugation.models_v2 import V2ModelSpec
from railguard.corrugation.validation_v2 import build_v2_validation_report
from railguard.project_paths import participant_root

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = participant_root(ROOT) / "02_Datasets" / "Rail_Corrugation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "cache" / "corrugation_v2" / "train_features.csv"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "artifacts" / "corrugation_v2" / "model.joblib"
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT / "reports" / "corrugation_v2" / "training_report.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "reports" / "corrugation_v2" / "out_of_fold_predictions.csv",
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "corrugation_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    labels_frame = pd.read_csv(args.labels).set_index("filename")
    labels = labels_frame.loc[features["file_id"], "label"].to_numpy()
    predictions = pd.read_csv(args.predictions)
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    artifact = load_corrugation_artifact(args.model)
    config = CorrugationConfig(**artifact["metadata"]["config"])
    spec = V2ModelSpec(**artifact["metadata"]["final_spec"])
    report = build_v2_validation_report(
        features, labels, predictions, training_report, spec, config
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "robustness_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
