"""Evaluate macro-F1-first selection on labelled rail recordings only."""

from pathlib import Path

from railguard.corrugation.objective_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(Path(__file__).resolve().parents[1])
