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

def test_synchronized_targets_include_conservative_clock_uncertainty(monkeypatch):
    readings = {
        "cam-a": {"offset_ms": 10, "uncertainty_ms": 2.0},
        "cam-b": {"offset_ms": -5, "uncertainty_ms": 3.0},
    }
    monkeypatch.setattr(
        capture,
        "measure_device_clock_offset_ms",
        lambda device, return_details=False: readings[device],
    )

    result = capture.synchronized_device_targets(["cam-a", "cam-b"], lead_ms=5000)

    assert result["sync_quality_ms"] == 5.0
    assert result["clock_uncertainty_ms"] == {"cam-a": 2.0, "cam-b": 3.0}
    assert result["device_target_ms"]["cam-a"] - 10 == result["host_target_ms"]
    assert result["device_target_ms"]["cam-b"] + 5 == result["host_target_ms"]
