"""Inference wrapper for the experimental normalized sensor-view blend."""

from railguard.corrugation.ensemble_experiment import blend_scores
from railguard.corrugation.models import labels_from_scores


class FittedCorrugationEnsemble:
    def __init__(self, baseline, augmented, alignment, weight):
        self.baseline = baseline
        self.augmented = augmented
        self.alignment = alignment
        self.weight = weight
        self.spec = baseline.spec

    def decision_scores(self, frame):
        base = self.baseline.decision_scores(frame)
        if self.weight == 0:
            return base
        return tuple(
            blend_scores(base, self.augmented.decision_scores(frame), self.weight, self.alignment)
        )

    def predict(self, frame):
        return labels_from_scores(
            *self.decision_scores(frame), self.spec.threshold, self.spec.side_i_bias
        )
