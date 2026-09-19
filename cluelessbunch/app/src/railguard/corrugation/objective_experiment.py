"""Paired nested experiment: recall-constrained versus macro-F1-first selection."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.config import VALID_LABELS, CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import FittedCorrugationModel
from railguard.corrugation.parsing import load_training_manifest
from railguard.corrugation.training import (
    _candidate_specs,
    duplicate_groups,
    load_or_extract_training_features,
)
from railguard.project_paths import participant_root

POLICIES = ("recall_constrained", "macro_f1_first")
EXPANDED_POLICY = "macro_f1_expanded"


def threshold_options(truth, first, second, config, *, exact=False):
    """Score the original grid or every observed cutoff plus both end decisions."""
    truth_codes = np.asarray([VALID_LABELS.index(label) for label in truth])
    options = []
    for bias in config.side_i_bias_values:
        maximum = np.maximum(first + bias, second)
        thresholds = np.unique(
            np.concatenate(
                (
                    np.quantile(
                        maximum,
                        np.linspace(0, 1, min(config.threshold_grid_size, len(maximum) + 1)),
                    ),
                    [0.0, np.nextafter(np.min(maximum), -np.inf)],
                )
            )
        )
        if exact:
            # Retain legacy cutoffs so this is a true superset of the original search.
            thresholds = np.unique(
                np.concatenate((thresholds, maximum, [np.nextafter(np.max(maximum), np.inf)]))
            )
        side = np.where(first + bias >= second, 1, 2)
        prediction = np.where(maximum[None, :] >= thresholds[:, None], side, 0)
        recall = np.zeros((len(thresholds), 3))
        f1 = np.zeros_like(recall)
        for label in range(3):
            actual = truth_codes == label
            positive = prediction == label
            tp = np.sum(positive & actual, axis=1)
            predicted_count = positive.sum(axis=1)
            support = actual.sum()
            recall[:, label] = tp / max(int(support), 1)
            f1[:, label] = np.divide(
                2.0 * tp,
                predicted_count + support,
                out=np.zeros(len(thresholds)),
                where=predicted_count + support > 0,
            )
        for index, threshold in enumerate(thresholds):
            options.append(
                (
                    float(threshold),
                    float(bias),
                    float(f1[index].mean()),
                    float(min(recall[index, 1:])),
                    float(
                        recall[index, 1] >= config.minimum_side_i_recall
                        and recall[index, 2] >= config.minimum_side_ii_recall
                    ),
                )
            )
    return options


def select_policies(features, labels, groups, config, *, include_expanded=False):
    """Reuse identical inner model scores for both decision-selection policies."""
    splits = list(
        StratifiedGroupKFold(
            n_splits=config.inner_folds, shuffle=True, random_state=config.random_state
        ).split(np.zeros(len(labels)), labels, groups)
    )
    winners, keys = {}, {}
    policies = POLICIES + ((EXPANDED_POLICY,) if include_expanded else ())
    complexity = {"time": 0, "compact": 1, "all": 2}
    for spec in _candidate_specs(config):
        first, second = np.empty(len(labels)), np.empty(len(labels))
        for train, validation in splits:
            assert not set(groups[train]) & set(groups[validation])
            model = FittedCorrugationModel(spec, config.random_state).fit(
                features.iloc[train], labels[train]
            )
            first[validation], second[validation] = model.decision_scores(features.iloc[validation])
        options = threshold_options(labels, first, second, config)
        expanded_options = options
        if include_expanded:
            expanded_config = replace(config, side_i_bias_values=(-0.4, -0.2, 0.0, 0.2, 0.4))
            # Legacy candidates come first, preserving them on exact ranking ties.
            expanded_options = options + threshold_options(
                labels, first, second, expanded_config, exact=True
            )
        for policy in policies:
            # Keep tie-breaking identical; only remove recall-gate precedence.
            chosen = max(
                expanded_options if policy == EXPANDED_POLICY else options,
                key=lambda x: (
                    x[4] if policy == "recall_constrained" else 0.0,
                    x[2],
                    x[3],
                    -abs(x[1]),
                ),
            )
            threshold, bias, macro_f1, fault_recall, _ = chosen
            key = (
                macro_f1,
                fault_recall,
                float(not spec.include_raw_speed),
                -float(complexity[spec.feature_family]),
            )
            if policy not in keys or key > keys[policy]:
                keys[policy] = key
                winners[policy] = replace(spec, threshold=threshold, side_i_bias=bias)
    return winners


def grouped_bootstrap(predictions, iterations=2000, seed=42, policies=POLICIES):
    """Resample duplicate groups once per draw, retaining every repeat for each group."""
    groups = predictions["group"].unique()
    group_indices = pd.Index(groups).get_indexer(predictions["group"])
    rng = np.random.default_rng(seed)
    # Store confusion contributions per group and repeat, so repeats stay dependent.
    repeats = sorted(predictions["repeat"].unique())
    contributions = np.zeros((2, len(repeats), len(groups), 3, 3), dtype=int)
    if len(policies) != 2:
        raise ValueError("Paired bootstrap requires exactly two policy columns")
    for p, policy in enumerate(policies):
        for r, repeat in enumerate(repeats):
            mask = predictions["repeat"].to_numpy() == repeat
            actual = np.array([VALID_LABELS.index(x) for x in predictions.loc[mask, "truth"]])
            predicted = np.array([VALID_LABELS.index(x) for x in predictions.loc[mask, policy]])
            np.add.at(contributions[p, r], (group_indices[mask], actual, predicted), 1)
    deltas = []
    for _ in range(iterations):
        draw = rng.integers(0, len(groups), len(groups))
        matrices = contributions[:, :, draw].sum(axis=2)
        tp = np.diagonal(matrices, axis1=-2, axis2=-1)
        denominator = matrices.sum(axis=-2) + matrices.sum(axis=-1)
        f1 = (
            np.divide(
                2.0 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
            )
            .mean(axis=-1)
            .mean(axis=-1)
        )
        deltas.append(f1[1] - f1[0])
    return {
        "unit": "duplicate group; all repeat predictions retained together",
        "statistic": "mean complete-repeat macro-F1 delta",
        "iterations": iterations,
        "median_delta": float(np.median(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "fraction_positive_deltas": float(np.mean(np.asarray(deltas) > 0)),
    }


def run_experiment(root: Path, config: CorrugationConfig | None = None, *, include_expanded=False):
    config = config or CorrugationConfig()
    data = participant_root(root) / "02_Datasets/Rail_Corrugation"
    policies = POLICIES + ((EXPANDED_POLICY,) if include_expanded else ())
    report_dir = (
        root
        / "reports"
        / ("corrugation_thresholds" if include_expanded else "corrugation_objective")
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    features = load_or_extract_training_features(
        data / "Train", manifest, root / "cache/corrugation", config
    )
    labels = manifest["label"].to_numpy()
    groups = duplicate_groups(features)
    predictions, folds = [], []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=config.random_state + repeat
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            assert not set(groups[train]) & set(groups[validation])
            selected = select_policies(
                features.iloc[train].reset_index(drop=True),
                labels[train],
                groups[train],
                config,
                include_expanded=include_expanded,
            )
            row = {"repeat": repeat + 1, "fold": fold}
            outputs = {}
            for policy, spec in selected.items():
                model = FittedCorrugationModel(spec, config.random_state + repeat).fit(
                    features.iloc[train], labels[train]
                )
                outputs[policy] = model.predict(features.iloc[validation])
                row[policy] = score_corrugation(labels[validation], outputs[policy]).macro_f1
                row[policy + "_spec"] = json.dumps(asdict(spec))
            folds.append(row)
            for local, idx in enumerate(validation):
                predictions.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": features.iloc[idx]["file_id"],
                        "group": groups[idx],
                        "truth": labels[idx],
                        **{policy: outputs[policy][local] for policy in policies},
                    }
                )
            # Checkpoint results after every fold; never use them for inner selection.
            pd.DataFrame(predictions).to_csv(
                report_dir / "out_of_fold_predictions.csv", index=False
            )
            pd.DataFrame(folds).to_csv(report_dir / "fold_results.csv", index=False)
            print(
                f"Repeat {repeat + 1}/{config.outer_repeats}, fold {fold}: "
                + ", ".join(f"{p}={row[p]:.4f}" for p in policies),
                flush=True,
            )
    predictions, folds = pd.DataFrame(predictions), pd.DataFrame(folds)
    summaries, repeat_rows = {}, []
    for repeat, frame in predictions.groupby("repeat"):
        repeat_rows.append(
            {
                "repeat": int(repeat),
                **{
                    policy: score_corrugation(
                        frame.truth.to_numpy(), frame[policy].to_numpy()
                    ).macro_f1
                    for policy in policies
                },
            }
        )
    for policy in policies:
        summaries[policy] = {
            **score_corrugation(
                predictions.truth.to_numpy(), predictions[policy].to_numpy()
            ).as_dict(),
            "mean_fold_macro_f1": float(folds[policy].mean()),
            "mean_repeat_macro_f1": float(np.mean([r[policy] for r in repeat_rows])),
        }
    delta = (
        summaries[policies[-1]]["mean_fold_macro_f1"] - summaries[POLICIES[0]]["mean_fold_macro_f1"]
    )
    report = {
        "test_data_used": False,
        "config": asdict(config),
        "source_files": len(features),
        "duplicate_groups": len(set(groups)),
        "changed_factor": (
            "symmetric bias grid and every distinct inner-score cutoff; same SVM candidate grid"
            if include_expanded
            else "recall gate precedence in inner threshold selection only"
        ),
        "candidate_metrics": summaries,
        "per_repeat": repeat_rows,
        "mean_fold_delta": delta,
        "paired_group_bootstrap": grouped_bootstrap(
            predictions, policies=(POLICIES[0], policies[-1])
        ),
        "promotion_delta_gate": bool(delta >= 0.02),
        "decision": "needs_further_robustness_checks" if delta >= 0.02 else "keep_v1",
        "limitation": "Previously inspected outer folds are development evidence, not a fresh test.",
    }
    if include_expanded:
        report["expanded_search"] = {
            "bias_values": [-0.4, -0.2, 0.0, 0.2, 0.4],
            "thresholds": "legacy grid plus all distinct scores plus all-Normal endpoint",
            "tie_preference": "legacy candidate on identical selection keys",
        }
        report["incremental_mean_fold_delta"] = (
            summaries[EXPANDED_POLICY]["mean_fold_macro_f1"]
            - summaries[POLICIES[1]]["mean_fold_macro_f1"]
        )
        report["incremental_group_bootstrap"] = grouped_bootstrap(
            predictions, policies=(POLICIES[1], EXPANDED_POLICY)
        )
    (report_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report
