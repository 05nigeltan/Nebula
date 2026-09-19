"""Nested, train-only experiment blending full-sensor and sensor-view SVMs."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import FittedCorrugationModel, labels_from_scores
from railguard.corrugation.models_augmented import FittedAugmentedCorrugationModel
from railguard.corrugation.objective_experiment import grouped_bootstrap, threshold_options
from railguard.corrugation.parsing import load_training_manifest
from railguard.corrugation.training import (
    _candidate_specs,
    duplicate_groups,
    load_or_extract_training_features,
)
from railguard.corrugation.training_augmented import (
    _view_training_arrays,
    load_or_extract_training_views,
)
from railguard.project_paths import participant_root

POLICIES = ("macro_f1_first", "sensor_ensemble")
BLEND_WEIGHTS = (0.0, 0.1, 0.25)


def margin_alignment(reference, auxiliary):
    """Fit a shared-side affine scale using training-full-recording margins only.

    These are NOT probabilities. Pooling both sides preserves exchange symmetry.
    Constant auxiliary margins contribute the reference mean, not an unstable ratio.
    """
    reference, auxiliary = np.asarray(reference), np.asarray(auxiliary)
    if reference.shape != auxiliary.shape or not np.isfinite([reference, auxiliary]).all():
        raise ValueError("Alignment requires matching finite margin arrays")
    spread = float(auxiliary.std())
    scale = float(reference.std()) / spread if spread > 1e-12 else 0.0
    return scale, float(reference.mean() - scale * auxiliary.mean())


def blend_scores(reference, auxiliary, weight, alignment):
    if not 0 <= weight <= 1:
        raise ValueError("Blend weight must be in [0, 1]")
    reference = np.asarray(reference)
    if weight == 0:
        return reference.copy()  # Exact baseline, including floating-point arithmetic.
    scale, offset = alignment
    return (1 - weight) * reference + weight * (scale * np.asarray(auxiliary) + offset)


def fit_pair(frame, labels, view_arrays, spec, seed):
    baseline = FittedCorrugationModel(spec, seed).fit(frame, labels)
    augmented = FittedAugmentedCorrugationModel(spec, seed).fit(*view_arrays)
    alignment = margin_alignment(baseline.decision_scores(frame), augmented.decision_scores(frame))
    return baseline, augmented, alignment


def select_ensemble(frame, views, labels, groups, config):
    """Jointly choose a shared component spec and blend on inner OOF scores.

    No preselected component settings, calibrators, or outer predictions are imported.
    Shared hyperparameters deliberately bound the search instead of testing all pairs.
    """
    label_map = dict(zip(frame.file_id, labels, strict=True))
    splits = StratifiedGroupKFold(
        n_splits=config.inner_folds, shuffle=True, random_state=config.random_state
    ).split(np.zeros(len(labels)), labels, groups)
    prepared = []
    for train, validation in splits:
        assert not set(groups[train]) & set(groups[validation])
        arrays = _view_training_arrays(views, frame.iloc[train].file_id.to_numpy(), label_map)
        prepared.append((train, validation, arrays))
    winners, keys = {}, {}
    complexity = {"time": 0, "compact": 1, "all": 2}
    for spec in _candidate_specs(config):
        scores = {weight: np.empty((2, len(frame))) for weight in BLEND_WEIGHTS}
        for train, validation, arrays in prepared:
            first, second, alignment = fit_pair(
                frame.iloc[train], labels[train], arrays, spec, config.random_state
            )
            base_scores = first.decision_scores(frame.iloc[validation])
            aug_scores = second.decision_scores(frame.iloc[validation])
            for weight in BLEND_WEIGHTS:
                scores[weight][:, validation] = blend_scores(
                    base_scores, aug_scores, weight, alignment
                )
        for weight in BLEND_WEIGHTS:
            choice = max(
                threshold_options(labels, *scores[weight], config),
                key=lambda x: (x[2], x[3], -abs(x[1])),
            )
            threshold, bias, f1, recall, _ = choice
            key = (
                f1,
                recall,
                -weight,
                float(not spec.include_raw_speed),
                -float(complexity[spec.feature_family]),
            )
            for policy in POLICIES if weight == 0 else (POLICIES[1],):
                if policy not in keys or key > keys[policy]:
                    keys[policy] = key
                    winners[policy] = (replace(spec, threshold=threshold, side_i_bias=bias), weight)
    return winners


def evaluate(frame, views, labels, groups, train, validation, config, seed):
    assert not set(groups[train]) & set(groups[validation])
    train_frame = frame.iloc[train].reset_index(drop=True)
    selected = select_ensemble(train_frame, views, labels[train], groups[train], config)
    label_map = dict(zip(train_frame.file_id, labels[train], strict=True))
    arrays = _view_training_arrays(views, train_frame.file_id.to_numpy(), label_map)
    outputs, specs = {}, {}
    for policy, (spec, weight) in selected.items():
        baseline = FittedCorrugationModel(spec, seed).fit(train_frame, labels[train])
        scores = baseline.decision_scores(frame.iloc[validation])
        if weight:
            augmented = FittedAugmentedCorrugationModel(spec, seed).fit(*arrays)
            alignment = margin_alignment(
                baseline.decision_scores(train_frame), augmented.decision_scores(train_frame)
            )
            scores = blend_scores(
                scores, augmented.decision_scores(frame.iloc[validation]), weight, alignment
            )
        outputs[policy] = labels_from_scores(*scores, spec.threshold, spec.side_i_bias)
        specs[policy] = {**asdict(spec), "augmented_weight": weight}
    return outputs, specs


def run_ensemble_experiment(root: Path, config=None):
    config = config or CorrugationConfig()
    data = participant_root(root) / "02_Datasets/Rail_Corrugation"
    report_dir = root / "reports/corrugation_ensemble"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    frame = load_or_extract_training_features(
        data / "Train", manifest, root / "cache/corrugation", config
    )
    full, views = load_or_extract_training_views(
        data / "Train", manifest, root / "cache/corrugation_aug", config
    )
    if (
        frame.file_id.tolist() != full.file_id.tolist()
        or frame.sha256.tolist() != full.sha256.tolist()
    ):
        raise ValueError("Full and augmentation source caches disagree")
    labels, groups = manifest.label.to_numpy(), duplicate_groups(frame)
    folds, predictions = [], []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=config.random_state + repeat
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            outputs, specs = evaluate(
                frame,
                views,
                labels,
                groups,
                train,
                validation,
                config,
                config.random_state + repeat,
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
                        "file_id": frame.iloc[idx].file_id,
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
                + ", ".join(f"{p}={row[p]:.4f}" for p in POLICIES)
                + f", weight={specs[POLICIES[1]]['augmented_weight']}",
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
    delta = metrics[POLICIES[1]]["mean_fold_macro_f1"] - metrics[POLICIES[0]]["mean_fold_macro_f1"]
    report = {
        "test_data_used": False,
        "deployment_changed": False,
        "config": asdict(config),
        "source_files": len(frame),
        "duplicate_groups": len(set(groups)),
        "blend_weights": BLEND_WEIGHTS,
        "shared_component_spec": True,
        "normalization": "Affine match of pooled-side training-full margins to baseline scale",
        "candidate_metrics": metrics,
        "per_repeat": repeats,
        "incremental_mean_fold_delta": delta,
        "paired_group_bootstrap": grouped_bootstrap(predictions, policies=POLICIES),
        "selected_weight_counts": pd.Series(
            [json.loads(s)["augmented_weight"] for s in folds.sensor_ensemble_spec]
        )
        .value_counts()
        .to_dict(),
        "limitation": "Previously inspected development folds, not an independent final test.",
        "decision": "requires_further_review" if delta > 0 else "reject_ensemble",
    }
    path = report_dir / "comparison_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    speed_rows = []
    speed = frame.speed_mps.to_numpy()
    for lower, upper in ((9.5, 12.0), (12.0, 15.0), (15.0, np.inf)):
        mask = (speed >= lower) & (speed < upper)
        validation = np.flatnonzero(mask)
        train = np.flatnonzero(~mask & ~np.isin(groups, groups[validation]))
        outputs, specs = evaluate(
            frame, views, labels, groups, train, validation, config, config.random_state
        )
        speed_rows.append(
            {
                "lower_mps": lower,
                "upper_mps": upper if np.isfinite(upper) else None,
                "class_counts": pd.Series(labels[validation]).value_counts().to_dict(),
                "metrics": {
                    p: score_corrugation(labels[validation], outputs[p]).as_dict() for p in POLICIES
                },
                "selected_specs": specs,
            }
        )
        print(f"Completed speed holdout {lower} to {upper} m/s", flush=True)
    report["speed_holdouts"] = speed_rows
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "delta": delta}, indent=2), flush=True)
    return report
