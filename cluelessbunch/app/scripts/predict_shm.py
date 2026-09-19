"""Generate an SHM submission CSV from a directory of stress-signal CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from railguard.shm.inference import predict_shm_directory, write_predictions_atomic

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory of SHM CSV files")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV to create")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "shm" / "model.joblib",
        help="Trained SHM artifact",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional CSV for non-submission input diagnostics",
    )
    parser.add_argument(
        "--training-features",
        type=Path,
        default=ROOT / "cache" / "shm" / "train_features.csv",
        help="Feature cache used only to flag diagnostic range departures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, diagnostics = predict_shm_directory(args.input, args.model)
    write_predictions_atomic(predictions, args.output)
    if args.diagnostics is not None:
        if args.training_features.is_file():
            training = pd.read_csv(args.training_features)
            mapping = {
                "signal_rms": "stat_rms",
                "maximum_cycle_range": "rf_max_range",
                "top_1pct_damage_share": "rf_top_01_damage_share",
            }
            flag_columns = []
            for diagnostic, feature in mapping.items():
                flag = f"{diagnostic}_outside_training_range"
                diagnostics[flag] = (diagnostics[diagnostic] < training[feature].min()) | (
                    diagnostics[diagnostic] > training[feature].max()
                )
                flag_columns.append(flag)
            diagnostics["outside_training_range_fields"] = diagnostics[flag_columns].sum(axis=1)
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.diagnostics, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(predictions),
                "prediction_min": float(predictions["prediction"].min()),
                "prediction_max": float(predictions["prediction"].max()),
                "maximum_correction_from_physics": float(
                    np.max(np.abs(diagnostics["correction_factor"] - 1.0))
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
