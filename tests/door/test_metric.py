import pandas as pd

from railguard.door.metric import score_door_segments


def test_official_metric_is_one_for_exact_predictions() -> None:
    truth = pd.DataFrame(
        [{"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-1-0", "status": "Normal"}]
    )
    predicted = truth.rename(columns={"status": "prediction"})
    score = score_door_segments(truth, predicted)
    assert score.score == 1.0
    assert score.matches == 1


def test_wrong_label_gets_no_credit() -> None:
    truth = pd.DataFrame(
        [{"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-1-0", "status": "Normal"}]
    )
    predicted = truth.rename(columns={"status": "prediction"})
    predicted["prediction"] = "Abnormal resistance"
    assert score_door_segments(truth, predicted).score == 0.0


def test_half_overlap_receives_half_iou_credit() -> None:
    truth = pd.DataFrame(
        [{"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-2-0", "status": "Normal"}]
    )
    predicted = pd.DataFrame(
        [{"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-1-0", "prediction": "Normal"}]
    )
    assert score_door_segments(truth, predicted).score == 0.5
