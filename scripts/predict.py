"""Generate a Door submission CSV from a continuous sensor stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.door.inference import predict_door

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Continuous Door CSV/Excel file")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV to create")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "door" / "model.joblib",
        help="Trained Door artifact",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel worksheet name or zero-based index (default: first worksheet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheet_name = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    predictions, diagnostics = predict_door(
        args.input,
        args.model,
        sheet_name=sheet_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "segments": len(predictions),
                "segmentation": diagnostics.__dict__,
                "label_counts": predictions["prediction"].value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
