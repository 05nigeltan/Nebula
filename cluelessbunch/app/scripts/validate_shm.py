"""Run additional train-only robustness checks for the SHM model decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from railguard.project_paths import participant_root
from railguard.shm.config import ShmConfig
from railguard.shm.robustness import (
    condition_cluster_validation,
    extreme_holdouts,
    paired_bootstrap,
    repeated_stratified_validation,
    residual_ablation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = participant_root(ROOT) / "02_Datasets" / "SHM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "cache" / "shm" / "train_features.csv"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA / "Train_Labels.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "shm")
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels).set_index("filename")
    target = labels.loc[features["file_id"], "damage"].to_numpy(float)
    config = ShmConfig()
    repeated, repeated_predictions = repeated_stratified_validation(
        features, target, config, repeats=args.repeats
    )
    clusters, cluster_predictions = condition_cluster_validation(features, target, config)
    extremes = extreme_holdouts(features, target, config)
    nested = pd.read_csv(args.report_dir / "out_of_fold_predictions.csv")
    bootstrap = paired_bootstrap(
        nested["truth"].to_numpy(float),
        nested["physics_prediction"].to_numpy(float),
        nested["hybrid_prediction"].to_numpy(float),
        config.random_state,
    )
    ablation = residual_ablation(features, target, config)
    robustness_supports_hybrid = bool(
        repeated["relative_mape_improvement"] >= config.hybrid_min_relative_improvement
        and repeated["sample_fold_win_rate"] >= config.hybrid_min_fold_win_rate
        and clusters["hybrid"]["mape"] <= clusters["physics"]["mape"]
        and bootstrap["ci95_lower"] > 0
    )
    report = {
        "test_data_used": False,
        "repeated_stratified_8fold": repeated,
        "condition_cluster_holdout": clusters,
        "extreme_target_holdouts": extremes,
        "nested_loo_paired_bootstrap": bootstrap,
        "residual_feature_ablation": ablation,
        "robustness_supports_hybrid": robustness_supports_hybrid,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "robustness_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    repeated_predictions.to_csv(
        args.report_dir / "repeated_validation_predictions.csv", index=False
    )
    cluster_predictions.to_csv(args.report_dir / "condition_cluster_predictions.csv", index=False)
    pd.DataFrame(ablation).to_csv(args.report_dir / "ablation_results.csv", index=False)
    print(
        json.dumps(
            {
                "repeated_physics_mape": repeated["physics"]["mape"],
                "repeated_hybrid_mape": repeated["hybrid"]["mape"],
                "repeated_hybrid_win_rate": repeated["sample_fold_win_rate"],
                "cluster_physics_mape": clusters["physics"]["mape"],
                "cluster_hybrid_mape": clusters["hybrid"]["mape"],
                "bootstrap_gain_ci95": [bootstrap["ci95_lower"], bootstrap["ci95_upper"]],
                "robustness_supports_hybrid": robustness_supports_hybrid,
                "test_data_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
