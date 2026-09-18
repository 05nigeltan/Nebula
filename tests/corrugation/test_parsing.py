from railguard.corrugation.parsing import load_corrugation_file


def test_parser_maps_all_channels_and_sides(corrugation_csv) -> None:
    signal = load_corrugation_file(corrugation_csv, expected_samples=256)
    assert signal.signals.shape == (256, 128)
    assert len(signal.channels) == 128
    assert sum(channel.side == "side_i" for channel in signal.channels) == 64
    assert sum(channel.side == "side_ii" for channel in signal.channels) == 64
    assert sum(channel.signal_type == "vibration" for channel in signal.channels) == 64
    assert sum(channel.signal_type == "shock" for channel in signal.channels) == 64

