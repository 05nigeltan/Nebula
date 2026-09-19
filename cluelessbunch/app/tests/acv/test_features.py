from railguard.acv.config import AcvConfig
from railguard.acv.features import extract_case_features
from railguard.acv.models import FixedPhysicsRanker
from railguard.acv.parsing import load_acv_case


def test_peer_features_localise_hot_faulty_car(acv_workbook) -> None:
    features = extract_case_features(load_acv_case(acv_workbook), AcvConfig())
    faulty = features.set_index("car").loc["03"]
    assert faulty["hottest_fraction"] > 0.99
    assert faulty["peer_temp_q90"] > 3.0
    assert FixedPhysicsRanker().rank(features)[0] == "03"
