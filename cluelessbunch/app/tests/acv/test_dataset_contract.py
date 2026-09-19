from pathlib import Path

from railguard.acv.parsing import load_training_manifest
from railguard.project_paths import participant_root

ROOT = Path(__file__).resolve().parents[2]
DATA = participant_root(ROOT) / "02_Datasets" / "ACV"


def test_supplied_training_manifest_is_complete() -> None:
    manifest = load_training_manifest(DATA / "Train", DATA / "Train_Labels.csv")
    assert len(manifest) == 6
    assert manifest["faulty_car"].tolist() == ["01", "02", "03", "01", "04", "06"]
