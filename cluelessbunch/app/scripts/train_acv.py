"""Train and validate the ACV car-ranking model without reading competition test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.acv.training import train_and_save
from railguard.project_paths import participant_root

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = participant_root(ROOT) / "02_Datasets" / "ACV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA / "Train")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "acv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "acv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save(
        args.train_dir,
        args.labels,
        args.artifact_dir,
        args.report_dir,
    )
    print(
        json.dumps(
            {
                "artifact": result["artifact_path"],
                "selected_model": result["selected_model"],
                "candidate_metrics": result["candidate_metrics"],
                "robustness_metrics": result["robustness_metrics"],
                "training_files": result["training_files"],
                "test_data_used": result["test_data_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
