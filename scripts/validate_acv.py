"""Re-run train-only ACV validation and write a fresh validation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from railguard.acv.config import AcvConfig
from railguard.acv.features import extract_case_features
from railguard.acv.parsing import load_acv_case, load_training_manifest
from railguard.acv.validation import compare_models, feature_ablation, robustness_suite

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "ACV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA / "Train")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "acv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AcvConfig()
    manifest = load_training_manifest(args.train_dir, args.labels)
    faulty_by_file = dict(zip(manifest["filename"], manifest["faulty_car"], strict=True))
    cases = {
        filename: load_acv_case(args.train_dir / filename) for filename in manifest["filename"]
    }
    features = pd.concat(
        [extract_case_features(case, config) for case in cases.values()],
        ignore_index=True,
    )
    comparison = compare_models(features, faulty_by_file, config)
    ablations = feature_ablation(features, faulty_by_file)
    robustness = robustness_suite(cases, faulty_by_file, config)
    report = {
        "test_data_used": False,
        "selected_model": comparison["selected_model"],
        "fixed_metrics": comparison["fixed_metrics"],
        "learned_metrics": comparison["learned_metrics"],
        "selection_gate": {
            "score_gain": comparison["score_gain"],
            "learned_accepted": comparison["learned_accepted"],
        },
        "robustness_metrics": {
            "temporal_blocks": robustness["temporal_metrics"],
            "row_dropout": robustness["dropout_metrics"],
        },
        "ablations": ablations,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / "post_training_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
