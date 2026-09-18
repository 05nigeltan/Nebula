"""Run train-only robustness checks for the fitted Rail Corrugation specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.inference import load_corrugation_artifact
from railguard.corrugation.models import ModelSpec
from railguard.corrugation.validation import run_robustness_suite

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "NebulaX-Hackathon-ProblemStatement"
    / "PS3"
    / "02_Datasets"
    / "Rail_Corrugation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "cache" / "corrugation" / "train_features.csv"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "artifacts" / "corrugation" / "model.joblib"
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "corrugation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels).set_index("filename")
    target = labels.loc[features["file_id"], "label"].to_numpy()
    artifact = load_corrugation_artifact(args.model)
    config = CorrugationConfig(**artifact["metadata"]["config"])
    spec = ModelSpec(**artifact["metadata"]["final_spec"])
    report = run_robustness_suite(features, target, spec, config)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "robustness_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overlap_macro_f1": report["overlap_ge_9_5_mps"][
                    "fold_macro_f1_mean"
                ],
                "no_raw_speed_macro_f1": report["no_raw_speed"]["fold_macro_f1_mean"],
                "robustness_passed": report["acceptance"]["passed"],
                "test_data_used": report["test_data_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

