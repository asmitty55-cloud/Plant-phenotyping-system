from pt.device.capture_service import capture


def test_parse_device_uptime_ms():
    assert capture._parse_device_uptime_ms("123.456 789.000\n") == 123456


def test_measure_device_clock_offset_uses_lowest_rtt_samples(monkeypatch):
    host_times = iter([
        100_000_000_000, 100_010_000_000,
        200_000_000_000, 200_002_000_000,
        300_000_000_000, 300_003_000_000,
    ])
    device_values = iter(["100.005 0", "200.006 0", "300.008 0"])
    monkeypatch.setattr(capture.time, "monotonic_ns", lambda: next(host_times))
    monkeypatch.setattr(capture, "adb", lambda *args, **kwargs: (next(device_values), ""))

    # Midpoint offsets are 0, 5, and 6.5 ms; median of the three rounds to 5 ms.
    assert capture.measure_device_clock_offset_ms("phone", samples=3) == 5
