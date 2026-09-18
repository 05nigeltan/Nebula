"""Generate rail_predictions.csv from a directory of Corrugation CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.corrugation.inference import (
    predict_corrugation_directory,
    write_predictions_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory of Corrugation CSVs")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV to create")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "corrugation" / "model.joblib",
    )
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, diagnostics = predict_corrugation_directory(args.input, args.model)
    write_predictions_atomic(predictions, args.output)
    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.diagnostics, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(predictions),
                "class_counts": predictions["prediction"].value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

