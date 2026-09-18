from pathlib import Path

import pandas as pd

from railguard.door.config import DoorConfig
from railguard.door.training import load_labelled_cycles

ROOT = Path(__file__).resolve().parents[2]
DOOR_DATA = ROOT / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Door"


def test_supplied_training_stream_matches_all_labels() -> None:
    segments, targets, truth, diagnostics = load_labelled_cycles(
        DOOR_DATA / "Train.csv",
        DOOR_DATA / "Train_Segments_Answer.csv",
        DoorConfig(),
    )
    labels = pd.read_csv(DOOR_DATA / "Train_Segments_Answer.csv")
    assert len(segments) == len(labels) == diagnostics.cycle_count
    assert len(targets) == len(truth)
    assert set(targets) == {0, 1}
