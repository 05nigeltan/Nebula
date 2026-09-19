import pytest

from railguard.runtime import run_analysis
from railguard.uploads import UploadDataError


def test_inference_guard_rejects_overlap_and_recovers():
    with pytest.raises(UploadDataError, match="Another analysis"):
        run_analysis(lambda: run_analysis(lambda: 1))
    assert run_analysis(lambda x: x + 1, 4) == 5
    with pytest.raises(ValueError):
        run_analysis(lambda: int("not an integer"))
    assert run_analysis(lambda: "ready") == "ready"
