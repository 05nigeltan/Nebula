import pandas as pd

from railguard.operator_ui import (
    build_acv_card,
    build_corrugation_card,
    build_door_card,
    build_shm_card,
)


def test_door_card_turns_flags_into_an_action() -> None:
    predictions = pd.DataFrame(
        {
            "start_time": ["t0", "t1"],
            "end_time": ["t0e", "t1e"],
            "prediction": ["Normal", "Abnormal resistance"],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "start_time": ["t0", "t1"],
            "end_time": ["t0e", "t1e"],
            "operation": ["Open", "Close"],
            "score_margin": [-1.0, 1.2],
            "current_mean": [100.0, 700.0],
            "operation_warning": [None, None],
        }
    )
    card = build_door_card("door.csv", predictions, diagnostics)
    assert card.tone == "review"
    assert "1 of 2" in card.headline
    assert "Inspect" in card.action
    assert "detection boundary" not in card.evidence
    assert "different enough" in card.evidence


def test_acv_card_does_not_describe_rank_as_probability() -> None:
    diagnostics = pd.DataFrame(
        {
            "car": ["01", "02"],
            "predicted_rank": [1, 2],
            "supported": [True, True],
            "top_score_margin": [0.2, 0.2],
            "evidence_separation": ["Clear lead", "Clear lead"],
            "usable_samples": [100, 100],
            "usable_fraction": [0.8, 0.8],
            "hottest_fraction": [0.6, 0.4],
            "above_setpoint_fraction": [0.5, 0.2],
            "peer_temp_q90": [2.1, 0.1],
        }
    )
    card = build_acv_card("acv.xlsx", diagnostics)
    assert "Car 01" in card.headline
    assert "not proof" in card.limitation
    assert "probability" in card.limitation
    assert "score gap" not in card.evidence
    assert "stands out" in card.evidence


def test_corrugation_card_exposes_missing_location_mapping() -> None:
    card = build_corrugation_card(
        "recording.csv",
        "Side I",
        {
            "wavelength_valid": True,
            "margin_above_threshold": 0.4,
            "side_i_adjusted_score": 1.0,
            "side_ii_score": -0.2,
        },
    )
    assert card.tone == "review"
    assert "location" in card.limitation
    assert "route metadata" not in card.action
    assert "route record" in card.action
    assert "detection boundary" not in card.evidence


def test_shm_card_uses_reference_as_context_not_safety_threshold() -> None:
    card = build_shm_card(
        "stress.csv",
        11.0,
        {
            "sample_count": 100,
            "maximum_cycle_range": 6.0,
            "top_1pct_damage_share": 0.7,
        },
        {"q75": 5.0, "q90": 10.0},
    )
    assert "highest levels" in card.evidence
    assert card.status == "COMPARE WITH EARLIER RECORDINGS"
    assert "fatigue-damage index: 11" in card.headline
    assert "remaining-life estimate" in card.limitation
    assert "upper quartile" not in " ".join(card.reasons)
    assert "stress changes" in " ".join(card.reasons)
    assert "data_quality" not in card.as_dict()
