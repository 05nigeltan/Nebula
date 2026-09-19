"""Bounded Side I source-weight experiment with grouped nested validation."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from railguard.corrugation.config import SIDE_I_LABEL, CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import (
    FittedCorrugationModel,
    build_paired_matrix,
    feature_bases,
    paired_targets,
)
from railguard.corrugation.objective_experiment import grouped_bootstrap, threshold_options
from railguard.corrugation.parsing import load_training_manifest
from railguard.corrugation.training import (
    _candidate_specs,
    duplicate_groups,
    load_or_extract_training_features,
)
from railguard.project_paths import participant_root

FACTORS = (1.0, 1.4, 1.7)
POLICIES = ("recall_constrained", "macro_f1_first", "macro_f1_weighted")


def source_weights(labels, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Source weight factor must be finite and positive")
    weights = np.where(np.asarray(labels) == SIDE_I_LABEL, factor, 1.0)
    return weights / weights.mean()


class SourceWeightedModel(FittedCorrugationModel):
    """Keep preprocessing unchanged; weight both paired rows of each source file."""

    def __init__(self, spec, random_state=42, source_factor=1.0):
        super().__init__(spec, random_state)
        self.source_factor = source_factor

    def fit(self, frame, labels):
        weights = source_weights(labels, self.source_factor)
        if self.source_factor == 1.0:
            # Exact no-treatment control, including the estimator solver path.
            return super().fit(frame, labels)
        self.feature_bases_ = feature_bases(frame, self.spec.feature_family)
        matrix = build_paired_matrix(frame, self.feature_bases_, self.spec.include_raw_speed)
        target = paired_targets(labels)
        paired_weights = np.repeat(weights, 2)
        mass = np.bincount(target, weights=paired_weights, minlength=2)
        if np.any(mass <= 0):
            raise ValueError("Both binary side targets are required")
        class_weight = {
            0: paired_weights.sum() / (2 * mass[0]),
            1: paired_weights.sum() / (2 * mass[1]) * self.spec.positive_weight_multiplier,
        }
        self.scaler_ = StandardScaler().fit(matrix)
        self.classifier_ = LinearSVC(
            C=self.spec.c_value,
            class_weight=class_weight,
            dual="auto",
            max_iter=20_000,
            random_state=self.random_state,
        ).fit(self.scaler_.transform(matrix), target, sample_weight=paired_weights)
        return self


def select_weight_policies(features, labels, groups, config):
    splits = list(
        StratifiedGroupKFold(
            n_splits=config.inner_folds,
            shuffle=True,
            random_state=config.random_state,
        ).split(np.zeros(len(labels)), labels, groups)
    )
    winners, keys = {}, {}
    complexity = {"time": 0, "compact": 1, "all": 2}
    for factor in FACTORS:
        for spec in _candidate_specs(config):
            first, second = np.empty(len(labels)), np.empty(len(labels))
            for train, validation in splits:
                assert not set(groups[train]) & set(groups[validation])
                model = SourceWeightedModel(spec, config.random_state, factor).fit(
                    features.iloc[train], labels[train]
                )
                first[validation], second[validation] = model.decision_scores(
                    features.iloc[validation]
                )
            options = threshold_options(labels, first, second, config)
            for policy in POLICIES if factor == 1.0 else (POLICIES[-1],):
                chosen = max(
                    options,
                    key=lambda x: (x[4] if policy == POLICIES[0] else 0.0, x[2], x[3], -abs(x[1])),
                )
                threshold, bias, macro_f1, fault_recall, _ = chosen
                key = (
                    macro_f1,
                    fault_recall,
                    float(not spec.include_raw_speed),
                    -float(complexity[spec.feature_family]),
                    -factor,
                )
                if policy not in keys or key > keys[policy]:
                    keys[policy] = key
                    winners[policy] = (replace(spec, threshold=threshold, side_i_bias=bias), factor)
    return winners


def evaluate_split(features, labels, groups, train, validation, config, seed):
    assert not set(groups[train]) & set(groups[validation])
    selected = select_weight_policies(
        features.iloc[train].reset_index(drop=True), labels[train], groups[train], config
    )
    outputs, specs = {}, {}
    for policy, (spec, factor) in selected.items():
        model = SourceWeightedModel(spec, seed, factor).fit(features.iloc[train], labels[train])
        outputs[policy] = model.predict(features.iloc[validation])
        specs[policy] = {**asdict(spec), "source_factor": factor}
    return outputs, specs


def run_weight_experiment(root: Path, config=None):
    config = config or CorrugationConfig()
    data = participant_root(root) / "02_Datasets/Rail_Corrugation"
    report_dir = root / "reports/corrugation_weights"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    features = load_or_extract_training_features(
        data / "Train", manifest, root / "cache/corrugation", config
    )
    labels = manifest.label.to_numpy()
    groups = duplicate_groups(features)
    folds, predictions = [], []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=config.random_state + repeat
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            outputs, specs = evaluate_split(
                features, labels, groups, train, validation, config, config.random_state + repeat
            )
            row = {"repeat": repeat + 1, "fold": fold}
            for policy in POLICIES:
                row[policy] = score_corrugation(labels[validation], outputs[policy]).macro_f1
                row[policy + "_spec"] = json.dumps(specs[policy])
            folds.append(row)
            for local, idx in enumerate(validation):
                predictions.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "file_id": features.iloc[idx].file_id,
                        "group": groups[idx],
                        "truth": labels[idx],
                        **{p: outputs[p][local] for p in POLICIES},
                    }
                )
            pd.DataFrame(folds).to_csv(report_dir / "fold_results.csv", index=False)
            pd.DataFrame(predictions).to_csv(
                report_dir / "out_of_fold_predictions.csv", index=False
            )
            print(
                f"Repeat {repeat + 1}/{config.outer_repeats}, fold {fold}: "
                + ", ".join(f"{p}={row[p]:.4f}" for p in POLICIES),
                flush=True,
            )
    folds, predictions = pd.DataFrame(folds), pd.DataFrame(predictions)
    repeats = [
        {
            "repeat": int(r),
            **{
                p: score_corrugation(f.truth.to_numpy(), f[p].to_numpy()).macro_f1 for p in POLICIES
            },
        }
        for r, f in predictions.groupby("repeat")
    ]
    metrics = {
        p: {
            **score_corrugation(predictions.truth.to_numpy(), predictions[p].to_numpy()).as_dict(),
            "mean_fold_macro_f1": float(folds[p].mean()),
            "mean_repeat_macro_f1": float(np.mean([r[p] for r in repeats])),
        }
        for p in POLICIES
    }
    report = {
        "test_data_used": False,
        "source_files": len(labels),
        "duplicate_groups": len(set(groups)),
        "config": asdict(config),
        "source_factor_grid": list(FACTORS),
        "candidate_metrics": metrics,
        "per_repeat": repeats,
        "incremental_group_bootstrap": grouped_bootstrap(predictions, policies=POLICIES[1:]),
        "baseline_group_bootstrap": grouped_bootstrap(
            predictions, policies=(POLICIES[0], POLICIES[2])
        ),
        "limitation": "Previously inspected folds; development evidence, not a fresh test.",
    }
    comparison_path = report_dir / "comparison_report.json"
    comparison_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    speed = features.speed_mps.to_numpy()
    speed_rows = []
    # Reselect every parameter within the remaining speeds, avoiding global-spec leakage.
    for lower, upper in ((9.5, 12.0), (12.0, 15.0), (15.0, np.inf)):
        mask = (speed >= lower) & (speed < upper)
        validation = np.flatnonzero(mask)
        train = np.flatnonzero(~mask & ~np.isin(groups, groups[validation]))
        outputs, specs = evaluate_split(
            features, labels, groups, train, validation, config, config.random_state
        )
        speed_rows.append(
            {
                "lower_mps": lower,
                "upper_mps": upper if np.isfinite(upper) else None,
                "files": len(validation),
                "class_counts": pd.Series(labels[validation]).value_counts().to_dict(),
                "metrics": {
                    p: score_corrugation(labels[validation], outputs[p]).as_dict() for p in POLICIES
                },
                "selected_specs": specs,
            }
        )
        print(f"Completed speed holdout {lower} to {upper} m/s", flush=True)
        (report_dir / "speed_holdouts.json").write_text(
            json.dumps(speed_rows, indent=2), encoding="utf-8"
        )
    report["speed_holdouts"] = speed_rows
    report["incremental_mean_fold_delta"] = (
        metrics[POLICIES[2]]["mean_fold_macro_f1"] - metrics[POLICIES[1]]["mean_fold_macro_f1"]
    )
    report["mean_fold_delta_vs_original"] = (
        metrics[POLICIES[2]]["mean_fold_macro_f1"] - metrics[POLICIES[0]]["mean_fold_macro_f1"]
    )
    report["replacement_delta_gate"] = report["mean_fold_delta_vs_original"] >= 0.02
    report["weighted_beats_control"] = report["incremental_mean_fold_delta"] > 0
    report["deployment_changed"] = False
    comparison_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_metrics": metrics,
                "incremental_delta": report["incremental_mean_fold_delta"],
            },
            indent=2,
        ),
        flush=True,
    )
    return report
