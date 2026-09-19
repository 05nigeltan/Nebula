import pandas as pd

from railguard.acv.models import LinearListwiseRanker


def test_listwise_ranker_learns_positive_signal() -> None:
    rows = []
    labels = {}
    for case, faulty in (("a.xlsx", "01"), ("b.xlsx", "02"), ("c.xlsx", "03")):
        labels[case] = faulty
        for car in ("01", "02", "03"):
            value = 1.0 if car == faulty else 0.0
            rows.append(
                {
                    "file_id": case,
                    "car": car,
                    "rank_hottest_fraction": value,
                    "rank_above_setpoint_fraction": value,
                    "rank_peer_temp_q90": value,
                    "rank_control_error_q90": value,
                }
            )
    frame = pd.DataFrame(rows)
    model = LinearListwiseRanker(l2=0.1).fit(frame, labels)
    assert model.rank(frame[frame["file_id"].eq("c.xlsx")])[0] == "03"
