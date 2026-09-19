import numpy as np
import pandas as pd
import pytest

from railguard.corrugation.config import VALID_LABELS, CorrugationConfig
from railguard.corrugation.models import FittedCorrugationModel, ModelSpec
from railguard.corrugation.weight_experiment import (
    SourceWeightedModel,
    select_weight_policies,
    source_weights,
)


def test_source_weights_preserve_mass_and_relative_cost():
    labels = np.array(["Normal", "Normal", "Side I", "Side II"])
    weights = source_weights(labels, 1.7)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[2] / weights[0] == pytest.approx(1.7)
    assert weights[3] == weights[0]
    with pytest.raises(ValueError):
        source_weights(labels, -1)


def test_weighted_model_control_equivalence_and_fixed_scaling():
    rng = np.random.default_rng(42)
    labels = np.tile(VALID_LABELS, 10)
    frame = pd.DataFrame(
        {
            "side_i_vibration_rms_median": rng.normal(size=30),
            "side_ii_vibration_rms_median": rng.normal(size=30),
            "wavelength_valid": np.ones(30),
        }
    )
    spec = ModelSpec("time", 0.01, 1.0, False)
    baseline = FittedCorrugationModel(spec).fit(frame, labels)
    control = SourceWeightedModel(spec).fit(frame, labels)
    weighted = SourceWeightedModel(spec, source_factor=1.7).fit(frame, labels)
    np.testing.assert_array_equal(baseline.decision_scores(frame), control.decision_scores(frame))
    np.testing.assert_array_equal(baseline.scaler_.mean_, weighted.scaler_.mean_)
    np.testing.assert_array_equal(baseline.scaler_.scale_, weighted.scaler_.scale_)
    assert not np.allclose(baseline.classifier_.coef_, weighted.classifier_.coef_)


def test_weight_selection_keeps_duplicate_groups_out_of_inner_training(monkeypatch):
    groups = np.repeat(np.arange(12), 2)
    labels = np.repeat(np.tile(VALID_LABELS, 4), 2)
    frame = pd.DataFrame({"group_id": groups})
    seen_factors = []

    class GuardedModel:
        def __init__(self, spec, seed, factor):
            seen_factors.append(factor)

        def fit(self, frame, labels):
            self.training_groups = set(frame.group_id)
            return self

        def decision_scores(self, frame):
            assert self.training_groups.isdisjoint(frame.group_id)
            return np.zeros(len(frame)), np.ones(len(frame))

    monkeypatch.setattr("railguard.corrugation.weight_experiment.SourceWeightedModel", GuardedModel)
    config = CorrugationConfig(
        inner_folds=2,
        feature_families=("time",),
        c_values=(0.01,),
        positive_weight_multipliers=(1.0,),
    )
    winners = select_weight_policies(frame, labels, groups, config)
    assert set(seen_factors) == {1.0, 1.4, 1.7}
    assert winners["macro_f1_weighted"][1] == 1.0  # Smaller factor wins identical scores.
