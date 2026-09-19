"""Test the small full-sensor/sensor-view ensemble without changing deployment."""

from pathlib import Path

from railguard.corrugation.ensemble_experiment import run_ensemble_experiment

if __name__ == "__main__":
    run_ensemble_experiment(Path(__file__).resolve().parents[1])
