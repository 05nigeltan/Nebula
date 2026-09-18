"""Complete-case model comparison and ACV robustness validation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from railguard.acv.config import CORE_FEATURES, AcvConfig
from railguard.acv.features import extract_case_features
from railguard.acv.metric import rank_decay_for_position, score_rank_positions
from railguard.acv.models import FixedPhysicsRanker, LinearListwiseRanker
from railguard.acv.parsing import AcvCase


def _result_row(
    file_id: str,
    faulty_car: str,
    ranking: list[str],
    model_name: str,
    **extra: Any,
) -> dict[str, Any]:
    rank = ranking.index(faulty_car) + 1
    return {
        "file_id": file_id,
        "model": model_name,
        "faulty_car": faulty_car,
        "true_rank": rank,
        "rank_decay_score": rank_decay_for_position(rank, len(ranking)),
        "ranked_cars": "|".join(ranking),
        **extra,
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    ranks = np.asarray([row["true_rank"] for row in rows], dtype=int)
    counts = np.asarray([len(str(row["ranked_cars"]).split("|")) for row in rows], dtype=int)
    return score_rank_positions(ranks, counts).as_dict()


def _select_l2(
    frame: pd.DataFrame,
    files: list[str],
    faulty_by_file: dict[str, str],
    config: AcvConfig,
) -> tuple[float, list[dict[str, Any]]]:
    candidate_rows = []
    best_key: tuple[float, float, float] | None = None
    best_l2 = config.linear_l2_values[-1]
    for l2 in config.linear_l2_values:
        ranks = []
        counts = []
        for held_out in files:
            train_files = [file_id for file_id in files if file_id != held_out]
            train = frame[frame["file_id"].isin(train_files)]
            validation = frame[frame["file_id"].eq(held_out)]
            model = LinearListwiseRanker(l2=l2).fit(train, faulty_by_file)
            ranking = model.rank(validation)
            ranks.append(ranking.index(faulty_by_file[held_out]) + 1)
            counts.append(len(ranking))
        metrics = score_rank_positions(np.asarray(ranks), np.asarray(counts))
        row = {"l2": l2, **metrics.as_dict()}
        candidate_rows.append(row)
        key = (metrics.mean_rank_decay, -float(metrics.worst_rank), l2)
        if best_key is None or key > best_key:
            best_key = key
            best_l2 = l2
    return best_l2, candidate_rows


def compare_models(
    features: pd.DataFrame,
    faulty_by_file: dict[str, str],
    config: AcvConfig,
) -> dict[str, Any]:
    """Compare fixed and learned models using nested leave-one-case-out validation."""

    ordinary_files = sorted(
        features.loc[features["schema"].eq("standard"), "file_id"].unique().tolist()
    )
    ordinary = features[features["file_id"].isin(ordinary_files)].copy()
    fixed = FixedPhysicsRanker()
    fixed_rows = [
        _result_row(
            file_id,
            faulty_by_file[file_id],
            fixed.rank(ordinary[ordinary["file_id"].eq(file_id)]),
            "peer_physics_fixed",
        )
        for file_id in ordinary_files
    ]

    learned_rows: list[dict[str, Any]] = []
    inner_candidates: list[dict[str, Any]] = []
    for held_out in ordinary_files:
        train_files = [file_id for file_id in ordinary_files if file_id != held_out]
        l2, candidates = _select_l2(ordinary, train_files, faulty_by_file, config)
        for row in candidates:
            inner_candidates.append({"outer_file": held_out, **row})
        model = LinearListwiseRanker(l2=l2).fit(
            ordinary[ordinary["file_id"].isin(train_files)], faulty_by_file
        )
        validation = ordinary[ordinary["file_id"].eq(held_out)]
        learned_rows.append(
            _result_row(
                held_out,
                faulty_by_file[held_out],
                model.rank(validation),
                "linear_listwise",
                selected_l2=l2,
            )
        )

    fixed_metrics = _summarise(fixed_rows)
    learned_metrics = _summarise(learned_rows)
    score_gain = float(
        learned_metrics["mean_rank_decay"] - fixed_metrics["mean_rank_decay"]
    )
    improves_worst = learned_metrics["worst_rank"] < fixed_metrics["worst_rank"]
    learned_accepted = bool(
        (score_gain >= config.learned_min_score_gain or improves_worst)
        and learned_metrics["top3_recall"] >= fixed_metrics["top3_recall"]
    )
    final_l2, final_candidates = _select_l2(
        ordinary, ordinary_files, faulty_by_file, config
    )
    return {
        "ordinary_files": ordinary_files,
        "fixed_rows": fixed_rows,
        "learned_rows": learned_rows,
        "inner_candidate_rows": inner_candidates,
        "final_candidate_rows": final_candidates,
        "fixed_metrics": fixed_metrics,
        "learned_metrics": learned_metrics,
        "score_gain": score_gain,
        "learned_accepted": learned_accepted,
        "selected_model": "linear_listwise" if learned_accepted else "peer_physics_fixed",
        "final_l2": final_l2,
    }


def feature_ablation(
    features: pd.DataFrame,
    faulty_by_file: dict[str, str],
) -> list[dict[str, Any]]:
    ordinary = features[features["schema"].eq("standard")]
    files = sorted(ordinary["file_id"].unique())
    variants: dict[str, tuple[str, ...]] = {
        "all_core": CORE_FEATURES,
        **{f"only_{feature}": (feature,) for feature in CORE_FEATURES},
        **{
            f"without_{excluded}": tuple(
                feature for feature in CORE_FEATURES if feature != excluded
            )
            for excluded in CORE_FEATURES
        },
    }
    rows = []
    for name, raw_features in variants.items():
        rank_features = tuple(f"rank_{feature}" for feature in raw_features)
        model = FixedPhysicsRanker(
            feature_names=rank_features,
            weights=tuple([1.0 / len(rank_features)] * len(rank_features)),
        )
        results = [
            _result_row(
                file_id,
                faulty_by_file[file_id],
                model.rank(ordinary[ordinary["file_id"].eq(file_id)]),
                name,
            )
            for file_id in files
        ]
        rows.append({"variant": name, **_summarise(results)})
    return rows


def robustness_suite(
    cases: dict[str, AcvCase],
    faulty_by_file: dict[str, str],
    config: AcvConfig,
) -> dict[str, Any]:
    """Challenge the fixed score with temporal blocks and deterministic row dropout."""

    model = FixedPhysicsRanker()
    ordinary_files = [
        file_id
        for file_id, case in cases.items()
        if extract_case_features(case, config)["schema"].iloc[0] == "standard"
    ]
    temporal_rows = []
    for file_id in sorted(ordinary_files):
        case = cases[file_id]
        for block, indices in enumerate(np.array_split(np.arange(case.sample_count), config.temporal_blocks)):
            block_features = extract_case_features(case.subset(indices), config)
            if not block_features["supported"].any():
                continue
            temporal_rows.append(
                _result_row(
                    file_id,
                    faulty_by_file[file_id],
                    model.rank(block_features),
                    "temporal_block",
                    block=block,
                )
            )

    rng = np.random.default_rng(config.random_state)
    dropout_rows = []
    for fraction in config.dropout_fractions:
        for repeat in range(config.dropout_repeats):
            for file_id in sorted(ordinary_files):
                case = cases[file_id]
                keep_count = max(1, round(case.sample_count * (1.0 - fraction)))
                indices = np.sort(rng.choice(case.sample_count, size=keep_count, replace=False))
                reduced = extract_case_features(case.subset(indices), config)
                dropout_rows.append(
                    _result_row(
                        file_id,
                        faulty_by_file[file_id],
                        model.rank(reduced),
                        "row_dropout",
                        dropout_fraction=fraction,
                        repeat=repeat,
                    )
                )
    return {
        "test_data_used": False,
        "temporal_metrics": _summarise(temporal_rows),
        "temporal_rows": temporal_rows,
        "dropout_metrics": _summarise(dropout_rows),
        "dropout_rows": dropout_rows,
        "config": asdict(config),
    }
