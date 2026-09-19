"""Train, compare, and save the Door subsystem model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.door.training import train_and_save
from railguard.project_paths import participant_root

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = participant_root(ROOT) / "02_Datasets" / "Door"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA / "Train.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Segments_Answer.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "door")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "door")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save(args.train, args.labels, args.artifact_dir, args.report_dir)
    selected = next(
        item for item in result["candidate_results"] if item["name"] == result["model_name"]
    )
    print(
        json.dumps(
            {
                "artifact": result["artifact_path"],
                "selected_model": result["model_name"],
                "out_of_fold_score": selected["official_score"],
                "out_of_fold_accuracy": selected["accuracy"],
                "abnormal_recall": selected["abnormal_recall"],
                "cycles": result["training_cycles"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
