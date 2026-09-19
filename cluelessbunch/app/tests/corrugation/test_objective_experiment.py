from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from railguard.corrugation.config import VALID_LABELS, CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import choose_threshold, labels_from_scores
from railguard.corrugation.objective_experiment import (
    grouped_bootstrap,
    select_policies,
    threshold_options,
)


@pytest.mark.parametrize("seed", range(10))
def test_batched_thresholds_match_existing_selector(seed):
    rng = np.random.default_rng(seed)
    labels = rng.choice(VALID_LABELS, 80, p=[0.8, 0.08, 0.12])
    first, second = rng.normal(size=(2, 80))
    config = CorrugationConfig()
    options = threshold_options(labels, first, second, config)
    for recall_gate in (True, False):
        threshold, bias, score = choose_threshold(
            labels,
            first,
            second,
            config.threshold_grid_size,
            config.minimum_side_i_recall if recall_gate else 0.0,
            config.minimum_side_ii_recall if recall_gate else 0.0,
            config.side_i_bias_values,
        )
        chosen = max(options, key=lambda x: (x[4] if recall_gate else 0.0, x[2], x[3], -abs(x[1])))
        assert chosen[:2] == (threshold, bias)
        assert chosen[2] == pytest.approx(score.macro_f1, abs=1e-14)


def test_inner_selection_never_scores_a_training_duplicate(monkeypatch):
    groups = np.repeat(np.arange(12), 2)
    labels = np.repeat(np.tile(VALID_LABELS, 4), 2)
    frame = pd.DataFrame({"group_id": groups})
    scored = []

    class GuardedEstimator:
        def __init__(self, *args):
            pass

        def fit(self, frame, labels):
            self.training_groups = set(frame.group_id)
            return self

        def decision_scores(self, frame):
            assert self.training_groups.isdisjoint(frame.group_id)
            scored.extend(frame.index)
            return np.zeros(len(frame)), np.ones(len(frame))

    monkeypatch.setattr(
        "railguard.corrugation.objective_experiment.FittedCorrugationModel", GuardedEstimator
    )
    config = CorrugationConfig(
        inner_folds=2,
        feature_families=("time",),
        c_values=(0.01,),
        positive_weight_multipliers=(1.0,),
    )
    select_policies(frame, labels, groups, config, include_expanded=True)
    assert sorted(scored) == list(range(len(frame)))


def test_bootstrap_keeps_repeated_observations_together():
    frame = pd.DataFrame(
        {
            "repeat": [1, 1, 1],
            "group": ["a", "b", "c"],
            "truth": VALID_LABELS,
            "recall_constrained": ["Normal", "Normal", "Side II"],
            "macro_f1_first": VALID_LABELS,
        }
    )
    repeated = pd.concat([frame, frame.assign(repeat=2)], ignore_index=True)
    once = grouped_bootstrap(frame, iterations=100)
    twice = grouped_bootstrap(repeated, iterations=100)
    assert once == twice
    assert once["median_delta"] > 0


def test_expanded_grid_covers_all_decisions_and_preserves_legacy_candidates():
    config = CorrugationConfig()
    labels = np.array(["Normal", "Side I", "Side II", "Normal", "Side I"])
    first = np.array([0.1, 0.1, -0.3, 0.6, 0.8])
    second = np.array([-0.1, 0.2, 0.7, 0.6, -0.2])
    expanded = replace(config, side_i_bias_values=(-0.4, -0.2, 0.0, 0.2, 0.4))
    legacy = threshold_options(labels, first, second, config)
    options = threshold_options(labels, first, second, expanded, exact=True)
    assert set(legacy).issubset(options)
    for threshold, bias, f1, _, _ in options:
        prediction = labels_from_scores(first, second, threshold, bias)
        assert f1 == pytest.approx(score_corrugation(labels, prediction).macro_f1)
    for bias in expanded.side_i_bias_values:
        grid = [x[0] for x in options if x[1] == bias]
        scores = np.maximum(first + bias, second)
        assert set(scores).issubset(grid)
        assert max(grid) > max(scores)
        assert min(grid) < min(scores)
