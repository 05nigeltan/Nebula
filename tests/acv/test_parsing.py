from railguard.acv.parsing import load_acv_case


def test_loads_car_ids_and_timestamps(acv_workbook) -> None:
    case = load_acv_case(acv_workbook)
    assert case.car_ids == ("01", "02", "03", "04")
    assert case.sample_count == 120
    assert case.timestamps.is_monotonic_increasing
