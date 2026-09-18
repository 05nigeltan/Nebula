import numpy as np
import pandas as pd

from railguard.acv.metric import rank_decay_for_position, score_rank_positions


def test_rank_decay_matches_competition_definition() -> None:
    assert rank_decay_for_position(1, 8) == 1.0
    assert rank_decay_for_position(3, 8) == 0.75
    score = score_rank_positions(np.array([1, 3]), np.array([8, 8]))
    assert score.mean_rank_decay == 0.875
    assert score.top1_accuracy == 0.5


def test_prediction_schema_example() -> None:
    from railguard.acv.metric import validate_prediction_frame

    validate_prediction_frame(
        pd.DataFrame({"file_id": ["case.xlsx"], "ranked_cars": ["03|01|02|04"]})
    )
