import numpy as np
import pandas as pd
import pytest

from railguard.corrugation.config import VALID_LABELS, CorrugationConfig
from railguard.corrugation.ensemble_experiment import (
    blend_scores,
    margin_alignment,
    select_ensemble,
)
from railguard.corrugation.objective_experiment import select_policies


def test_serialized_ensemble_matches_blend_and_zero_fallback(tmp_path):
    import joblib

    from railguard.corrugation.models import FittedCorrugationModel, ModelSpec, labels_from_scores
    from railguard.corrugation.models_ensemble import FittedCorrugationEnsemble

    frame = pd.DataFrame(
        {
            "side_i_vibration_rms_median": [0.1, 2.0, 0.2, 0.3, 1.8, 0.2],
            "side_ii_vibration_rms_median": [0.2, 0.1, 2.2, 0.1, 0.2, 1.9],
            "wavelength_valid": [1.0] * 6,
        }
    )
    labels = np.asarray(VALID_LABELS * 2)
    base = FittedCorrugationModel(ModelSpec("time", 0.01, 1.0, False)).fit(frame, labels)
    auxiliary = FittedCorrugationModel(ModelSpec("time", 0.03, 1.0, False)).fit(frame, labels)
    alignment = margin_alignment(base.decision_scores(frame), auxiliary.decision_scores(frame))
    model = FittedCorrugationEnsemble(base, auxiliary, alignment, 0.25)
    expected = blend_scores(
        base.decision_scores(frame), auxiliary.decision_scores(frame), 0.25, alignment
    )
    np.testing.assert_allclose(model.decision_scores(frame), expected)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(frame), labels_from_scores(*expected, 0, 0))
    zero = FittedCorrugationEnsemble(base, None, None, 0)
    np.testing.assert_array_equal(zero.predict(frame), base.predict(frame))


def test_alignment_scale_side_symmetry_and_exact_zero_fallback():
    base = np.array([[-1.0, 2.0, 3.0], [0.1, -0.5, 2.0]])
    auxiliary = 7 * base + 4
    alignment = margin_alignment(base, auxiliary)
    np.testing.assert_allclose(blend_scores(base, auxiliary, 0.25, alignment), base)
    np.testing.assert_array_equal(blend_scores(base, auxiliary, 0, alignment), base)
    assert margin_alignment(base[::-1], auxiliary[::-1]) == pytest.approx(alignment)
    assert margin_alignment(base, np.ones_like(base))[0] == 0
    with pytest.raises(ValueError):
        blend_scores(base, auxiliary, -0.1, alignment)
    with pytest.raises(ValueError):
        margin_alignment(base, auxiliary * np.nan)


def test_inner_view_isolation_normalization_and_zero_tie(monkeypatch):
    groups = np.repeat(np.arange(12), 2)
    labels = np.repeat(np.tile(VALID_LABELS, 4), 2)
    frame = pd.DataFrame({"group_id": groups, "file_id": [f"f{i}" for i in range(len(groups))]})
    config = CorrugationConfig(
        inner_folds=2,
        feature_families=("time",),
        c_values=(0.01,),
        positive_weight_multipliers=(1.0,),
    )
    align_calls = []

    class GuardedEstimator:
        def __init__(self, *args):
            pass

        def fit(self, data, truth, *weights):
            self.training_groups = set(data.group_id)
            return self

        def decision_scores(self, data):
            # Train-only scale fitting may score the whole training set; held-out
            # scoring must be completely disjoint (including duplicate groups).
            observed = set(data.group_id)
            assert observed == self.training_groups or observed.isdisjoint(self.training_groups)
            return np.zeros(len(data)), np.ones(len(data))

    def view_arrays(views, ids, label_map):
        selected = frame[frame.file_id.isin(ids)]
        assert set(selected.file_id) <= set(label_map)
        return selected, selected.file_id.map(label_map).to_numpy(), np.ones(len(selected))

    original_alignment = margin_alignment

    def checked_alignment(base, auxiliary):
        align_calls.append(len(base[0]))
        return original_alignment(base, auxiliary)

    module = "railguard.corrugation.ensemble_experiment."
    monkeypatch.setattr(module + "FittedCorrugationModel", GuardedEstimator)
    monkeypatch.setattr(module + "FittedAugmentedCorrugationModel", GuardedEstimator)
    monkeypatch.setattr(module + "_view_training_arrays", view_arrays)
    monkeypatch.setattr(module + "margin_alignment", checked_alignment)
    monkeypatch.setattr(
        "railguard.corrugation.objective_experiment.FittedCorrugationModel", GuardedEstimator
    )
    winners = select_ensemble(frame, pd.DataFrame(), labels, groups, config)
    control = select_policies(frame, labels, groups, config)["macro_f1_first"]
    assert winners["macro_f1_first"] == (control, 0.0)
    assert winners["sensor_ensemble"] == (control, 0.0)
    assert align_calls == [12, 12]
