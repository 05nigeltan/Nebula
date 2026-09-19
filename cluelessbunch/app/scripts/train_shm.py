"""Train and compare SHM damage models without reading competition test inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.project_paths import participant_root
from railguard.shm.training import train_and_save

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = participant_root(ROOT) / "02_Datasets" / "SHM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA / "Train")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "shm")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "shm")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "shm")
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
    selected = result["candidate_metrics"]["selected_model"]
    print(
        json.dumps(
            {
                "artifact": result["artifact_path"],
                "selected_model": result["selected_model"],
                "nested_leave_one_file_out_mape": selected["mape"],
                "estimated_official_score": selected["official_score"],
                "hybrid_accepted": result["hybrid_gate"]["accepted"],
                "training_files": result["training_files"],
                "test_data_used": result["test_data_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
