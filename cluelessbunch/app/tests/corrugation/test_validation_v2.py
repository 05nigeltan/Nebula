import pandas as pd

from railguard.corrugation.validation_v2 import paired_file_bootstrap


def test_paired_bootstrap_detects_consistently_better_predictions() -> None:
    rows = []
    truth = ["Normal", "Side I", "Side II"] * 4
    for repeat in (1, 2):
        for index, label in enumerate(truth):
            rows.append(
                {
                    "repeat": repeat,
                    "file_id": f"Train{index + 1}.csv",
                    "truth": label,
                    "baseline_prediction": "Normal",
                    "v2_prediction": label,
                }
            )
    result = paired_file_bootstrap(pd.DataFrame(rows), iterations=200, random_state=7)
    assert result["median_delta"] > 0
    assert result["probability_v2_better"] > 0.95
