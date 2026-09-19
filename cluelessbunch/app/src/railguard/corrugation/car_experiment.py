"""Isolated car-feature ablation with nested grouped selection and speed holdouts."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from railguard.corrugation.car_features import car_contrast_features, is_car_feature
from railguard.corrugation.config import CorrugationConfig
from railguard.corrugation.metric import score_corrugation
from railguard.corrugation.models import FittedCorrugationModel
from railguard.corrugation.objective_experiment import grouped_bootstrap, select_policies
from railguard.corrugation.parsing import load_corrugation_file, load_training_manifest
from railguard.corrugation.training import duplicate_groups, load_or_extract_training_features
from railguard.project_paths import participant_root

POLICIES = ("macro_f1_first", "car_aware")


def load_car_features(train_dir, base, cache_dir, config):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "train_car_features.csv"
    meta_path = cache_dir / "metadata.json"
    signature = {
        "schema": 1,
        "expected_samples": config.expected_samples,
        "sources": dict(zip(base.file_id, base.sha256, strict=True)),
    }
    if path.is_file() and meta_path.is_file():
        cached = pd.read_csv(path)
        if (
            json.loads(meta_path.read_text()) == signature
            and cached.file_id.tolist() == base.file_id.tolist()
        ):
            return pd.concat([base.reset_index(drop=True), cached.drop(columns="file_id")], axis=1)
    rows = []
    for index, row in enumerate(base.itertuples(), start=1):
        signal = load_corrugation_file(train_dir / row.file_id, config.expected_samples)
        if signal.sha256 != row.sha256:
            raise ValueError("Source changed since base feature verification")
        rows.append({"file_id": row.file_id, **car_contrast_features(signal)})
        if index % 25 == 0 or index == len(base):
            print(f"Extracted car features {index}/{len(base)}", flush=True)
    cached = pd.DataFrame(rows)
    cached.to_csv(path, index=False)
    meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return pd.concat([base.reset_index(drop=True), cached.drop(columns="file_id")], axis=1)


def evaluate(features, labels, groups, train, validation, config, seed):
    assert not set(groups[train]) & set(groups[validation])
    outputs, specs = {}, {}
    for policy in POLICIES:
        frame = (
            features
            if policy == "car_aware"
            else features.drop(columns=[c for c in features if is_car_feature(c)])
        )
        spec = select_policies(
            frame.iloc[train].reset_index(drop=True), labels[train], groups[train], config
        )["macro_f1_first"]
        model = FittedCorrugationModel(spec, seed).fit(frame.iloc[train], labels[train])
        outputs[policy] = model.predict(frame.iloc[validation])
        specs[policy] = asdict(spec)
    return outputs, specs


def run_car_experiment(root: Path, config=None):
    config = config or CorrugationConfig()
    data = participant_root(root) / "02_Datasets/Rail_Corrugation"
    report_dir = root / "reports/corrugation_cars"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_training_manifest(data / "Train", data / "Train_Labels.csv")
    base = load_or_extract_training_features(
        data / "Train", manifest, root / "cache/corrugation", config
    )
    features = load_car_features(data / "Train", base, root / "cache/corrugation_cars", config)
    labels, groups = manifest.label.to_numpy(), duplicate_groups(base)
    folds, predictions = [], []
    for repeat in range(config.outer_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=config.random_state + repeat
        )
        for fold, (train, validation) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            outputs, specs = evaluate(
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
                        "file_id": base.iloc[idx].file_id,
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
        "deployment_changed": False,
        "config": asdict(config),
        "added_features": [c for c in features if is_car_feature(c)],
        "candidate_metrics": metrics,
        "per_repeat": repeats,
        "incremental_mean_fold_delta": metrics[POLICIES[1]]["mean_fold_macro_f1"]
        - metrics[POLICIES[0]]["mean_fold_macro_f1"],
        "paired_group_bootstrap": grouped_bootstrap(predictions, policies=POLICIES),
        "limitation": "Previously inspected development folds; no independent final-test estimate.",
    }
    path = report_dir / "comparison_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    speed_rows = []
    speed = features.speed_mps.to_numpy()
    for lower, upper in ((9.5, 12.0), (12.0, 15.0), (15.0, np.inf)):
        mask = (speed >= lower) & (speed < upper)
        validation = np.flatnonzero(mask)
        train = np.flatnonzero(~mask & ~np.isin(groups, groups[validation]))
        outputs, specs = evaluate(
            features, labels, groups, train, validation, config, config.random_state
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
    report["decision"] = (
        "requires_further_review"
        if report["incremental_mean_fold_delta"] > 0
        else "reject_car_features"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({"metrics": metrics, "delta": report["incremental_mean_fold_delta"]}, indent=2),
        flush=True,
    )
    return report
