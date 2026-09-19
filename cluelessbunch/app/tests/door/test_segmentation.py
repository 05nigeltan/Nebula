from railguard.door.parsing import load_door_csv
from railguard.door.segmentation import segment_stream


def test_gap_segmentation_and_operation_inference(two_cycle_csv) -> None:
    segments, diagnostics = segment_stream(load_door_csv(two_cycle_csv))
    assert diagnostics.cycle_count == 2
    assert [segment.operation for segment in segments] == ["Open", "Close"]
    assert [segment.n_rows for segment in segments] == [3, 3]
    assert segments[0].start_time == "2023-7-5-0-0-0-0"
    assert segments[1].end_time == "2023-7-5-0-0-2-40"
