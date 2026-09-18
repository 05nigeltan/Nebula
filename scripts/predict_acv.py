"""Generate acv_predictions.csv from an ACV workbook or directory of workbooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.acv.inference import (
    predict_acv_directory,
    predict_acv_files,
    write_predictions_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ACV .xlsx file or directory")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV to create")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "acv" / "model.joblib",
    )
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.is_dir():
        predictions, diagnostics = predict_acv_directory(args.input, args.model)
    else:
        predictions, diagnostics = predict_acv_files([args.input], args.model)
    write_predictions_atomic(predictions, args.output)
    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.diagnostics, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(predictions),
                "rankings": predictions.set_index("file_id")["ranked_cars"].to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
