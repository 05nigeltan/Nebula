"""Test Side I source-class weighting and train-only speed holdouts."""

from pathlib import Path

from railguard.corrugation.weight_experiment import run_weight_experiment

if __name__ == "__main__":
    run_weight_experiment(Path(__file__).resolve().parents[1])
