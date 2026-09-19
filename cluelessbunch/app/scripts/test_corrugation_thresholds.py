"""Evaluate the predeclared broader decision-rule search on labelled rail data."""

from pathlib import Path

from railguard.corrugation.objective_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(Path(__file__).resolve().parents[1], include_expanded=True)
