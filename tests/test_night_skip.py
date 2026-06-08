from pt.api import app as api_app


def test_night_skip_tracks_first_last_and_daylight_timestamps(monkeypatch):
    saved = []
    created = []
    clock = iter([1000.0, 1180.0, 1360.0])

    monkeypatch.setattr(api_app.time, "time", lambda: next(clock))
    monkeypatch.setattr(api_app, "save_night_skip_state", lambda: saved.append(dict(api_app.night_skip_started)))

    def fake_placeholder(device_id, timestamp, duration_seconds, **metadata):
        created.append((device_id, timestamp, duration_seconds, metadata))
        return f"night_{timestamp}.jpg"

    monkeypatch.setattr(api_app, "create_night_placeholder", fake_placeholder)

    api_app.night_skip_started.clear()
    api_app.start_night_skip("cam1", "20260607_220000")
    api_app.start_night_skip("cam1", "20260607_230000")

    placeholder = api_app.finish_night_skip("cam1", daylight_timestamp="20260608_060000")

    assert placeholder == "night_20260608_020000.jpg"
    assert created == [
        (
            "cam1",
            "20260608_020000",
            8 * 60 * 60,
            {
                "first_timestamp": "20260607_220000",
                "last_timestamp": "20260607_230000",
                "daylight_timestamp": "20260608_060000",
            },
        )
    ]
    assert "cam1" not in api_app.night_skip_started
    assert saved[-1] == {}
