"""Train and validate the Rail Corrugation model without reading competition test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.corrugation.training import train_and_save

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
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA / "Train")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "corrugation")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "corrugation")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "corrugation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save(
        args.train_dir,
        args.labels,
        args.artifact_dir,
        args.report_dir,
        args.cache_dir,
    )
    selected = result["candidate_metrics"]["side_symmetric_linear_svm"]
    print(
        json.dumps(
            {
                "artifact": result["artifact_path"],
                "model": result["model_name"],
                "nested_macro_f1_mean": selected["fold_macro_f1_mean"],
                "nested_macro_f1_std": selected["fold_macro_f1_std"],
                "per_class_f1": selected["per_class_f1"],
                "final_spec": result["final_spec"],
                "pre_robustness_gates_passed": result[
                    "acceptance_gates_before_overlap_test"
                ]["passed"],
                "training_files": result["training_files"],
                "test_data_used": result["test_data_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

