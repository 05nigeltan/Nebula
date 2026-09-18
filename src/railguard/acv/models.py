"""Deterministic and regularized listwise ACV ranking models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from railguard.acv.config import CORE_FEATURES


def ranked_feature_names() -> tuple[str, ...]:
    return tuple(f"rank_{feature}" for feature in CORE_FEATURES)


def _natural_car_key(value: str) -> tuple[int, str]:
    return (int(value), value)


def ranking_from_scores(frame: pd.DataFrame, scores: np.ndarray) -> list[str]:
    scored = frame[["car"]].copy()
    scored["score"] = np.asarray(scores, dtype=float)
    scored["tie_key"] = scored["car"].map(_natural_car_key)
    return scored.sort_values(
        ["score", "tie_key"], ascending=[False, True], kind="mergesort"
    )["car"].tolist()


@dataclass(frozen=True)
class FixedPhysicsRanker:
    """Equal-weight peer/setpoint anomaly ranker."""

    feature_names: tuple[str, ...] = ranked_feature_names()
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[list(self.feature_names)].to_numpy(float)
        weights = np.asarray(self.weights, dtype=float)
        return matrix @ weights

    def rank(self, frame: pd.DataFrame) -> list[str]:
        return ranking_from_scores(frame, self.score(frame))


class LinearListwiseRanker:
    """Non-negative linear ranker fitted with grouped one-choice softmax loss."""

    def __init__(self, l2: float = 1.0, feature_names: tuple[str, ...] | None = None) -> None:
        self.l2 = float(l2)
        self.feature_names = feature_names or ranked_feature_names()

    def fit(self, frame: pd.DataFrame, faulty_by_file: dict[str, str]) -> LinearListwiseRanker:
        matrix = frame[list(self.feature_names)].to_numpy(float)
        groups = [
            np.flatnonzero(frame["file_id"].to_numpy() == file_id)
            for file_id in frame["file_id"].drop_duplicates()
        ]
        target_positions = []
        for indices in groups:
            file_id = str(frame.iloc[indices[0]]["file_id"])
            cars = frame.iloc[indices]["car"].astype(str).to_numpy()
            matches = np.flatnonzero(cars == faulty_by_file[file_id])
            if len(matches) != 1:
                raise ValueError(f"Faulty car is not represented exactly once in {file_id}")
            target_positions.append(int(matches[0]))

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            loss = 0.5 * self.l2 * float(weights @ weights)
            gradient = self.l2 * weights
            for indices, target_position in zip(groups, target_positions, strict=True):
                local = matrix[indices]
                local_scores = local @ weights
                shifted = local_scores - np.max(local_scores)
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum()
                loss += -float(local_scores[target_position]) + float(
                    np.log(np.exp(shifted).sum()) + np.max(local_scores)
                )
                gradient += probabilities @ local - local[target_position]
            return loss, gradient

        initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * matrix.shape[1],
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"ACV listwise optimization failed: {result.message}")
        total = float(np.sum(result.x))
        self.weights_ = result.x / total if total > 0 else initial
        self.optimizer_message_ = str(result.message)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("LinearListwiseRanker has not been fitted")
        return frame[list(self.feature_names)].to_numpy(float) @ self.weights_

    def rank(self, frame: pd.DataFrame) -> list[str]:
        return ranking_from_scores(frame, self.score(frame))
