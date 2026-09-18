"""ACV rank-decay scoring and submission-contract validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from railguard.acv.config import OUTPUT_COLUMNS
from railguard.acv.parsing import AcvDataError


@dataclass(frozen=True)
class AcvScore:
    mean_rank_decay: float
    top1_accuracy: float
    top3_recall: float
    mean_rank: float
    worst_rank: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_rank_decay": self.mean_rank_decay,
            "top1_accuracy": self.top1_accuracy,
            "top3_recall": self.top3_recall,
            "mean_rank": self.mean_rank,
            "worst_rank": self.worst_rank,
        }


def rank_decay_for_position(rank: int, car_count: int) -> float:
    if car_count < 1 or rank < 1 or rank > car_count:
        raise ValueError("Rank must be within the number of cars")
    return float((car_count - (rank - 1)) / car_count)


def score_rank_positions(ranks: np.ndarray, car_counts: np.ndarray) -> AcvScore:
    ranks = np.asarray(ranks, dtype=int)
    car_counts = np.asarray(car_counts, dtype=int)
    if ranks.shape != car_counts.shape or ranks.size == 0:
        raise ValueError("Ranks and car counts must be nonempty and aligned")
    scores = (car_counts - (ranks - 1)) / car_counts
    return AcvScore(
        mean_rank_decay=float(np.mean(scores)),
        top1_accuracy=float(np.mean(ranks == 1)),
        top3_recall=float(np.mean(ranks <= 3)),
        mean_rank=float(np.mean(ranks)),
        worst_rank=int(np.max(ranks)),
    )


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise AcvDataError(
            f"ACV predictions must have columns {OUTPUT_COLUMNS}, got {tuple(frame.columns)}"
        )
    if frame.empty:
        raise AcvDataError("ACV prediction output cannot be empty")
    if frame["file_id"].isna().any() or frame["file_id"].duplicated().any():
        raise AcvDataError("ACV prediction file IDs must be nonempty and unique")
    if not frame["file_id"].astype(str).str.lower().str.endswith(".xlsx").all():
        raise AcvDataError("Every ACV file_id must include its .xlsx extension")
    for ranking in frame["ranked_cars"]:
        if not isinstance(ranking, str) or not ranking:
            raise AcvDataError("Every ACV ranked_cars value must be a nonempty string")
        cars = ranking.split("|")
        if len(cars) != len(set(cars)):
            raise AcvDataError("A ranked_cars value cannot contain duplicate cars")
        if not all(len(car) == 2 and car.isdigit() for car in cars):
            raise AcvDataError("Every ranked car must be a two-digit identifier")

