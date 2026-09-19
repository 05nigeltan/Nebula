"""Evaluate compact car-aware features against the macro-F1-first control."""

from pathlib import Path

from railguard.corrugation.car_experiment import run_car_experiment

if __name__ == "__main__":
    run_car_experiment(Path(__file__).resolve().parents[1])
