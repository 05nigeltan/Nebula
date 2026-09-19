"""Train Rail Corrugation v2 without reading competition test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.corrugation.training_v2 import train_and_save_v2
from railguard.project_paths import participant_root

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = participant_root(ROOT) / "02_Datasets" / "Rail_Corrugation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA / "Train")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "corrugation_v2")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "corrugation_v2")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "corrugation_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save_v2(
        args.train_dir,
        args.labels,
        args.artifact_dir,
        args.report_dir,
        args.cache_dir,
    )
    metrics = result["candidate_metrics"]
    print(
        json.dumps(
            {
                "artifact": result["artifact_path"],
                "model": result["model_name"],
                "baseline_macro_f1_mean": metrics["baseline"]["fold_macro_f1_mean"],
                "v2_macro_f1_mean": metrics["v2"]["fold_macro_f1_mean"],
                "mean_fold_delta": metrics["v2"]["mean_fold_delta"],
                "v2_per_class_f1": metrics["v2"]["per_class_f1"],
                "final_spec": result["final_spec"],
                "pre_robustness_gates_passed": result["acceptance_gates_before_robustness"][
                    "passed"
                ],
                "test_data_used": result["test_data_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
