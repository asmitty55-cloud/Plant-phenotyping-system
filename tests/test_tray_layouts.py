from pt.core.analysis import tray_store


def test_radial_tray_layout_names_cells_by_position(monkeypatch, tmp_path):
    monkeypatch.setattr(tray_store, "TRAYS_PATH", str(tmp_path / "tray_cells.json"))

    tray = tray_store.upsert_tray({
        "tray_id": "blumat_trial",
        "name": "Blumat trial",
        "layout_type": "radial",
        "target_count": 8,
        "center_feature": "Blumat",
        "device_id": "escam",
    })
    cell = tray_store.add_cell({
        "tray_id": tray["tray_id"],
        "region": [10, 20, 40, 60],
        "position_index": 0,
    })

    assert tray["layout_type"] == "radial"
    assert tray["target_count"] == 8
    assert tray["center_feature"] == "Blumat"
    assert cell["cell_id"] == "blumat_trial_p1"
    assert cell["name"] == "P1"


def test_polygon_tray_cell_can_be_deleted(monkeypatch, tmp_path):
    monkeypatch.setattr(tray_store, "TRAYS_PATH", str(tmp_path / "tray_cells.json"))

    tray_store.upsert_tray({"tray_id": "blumat_trial", "layout_type": "radial"})
    cell = tray_store.add_cell({
        "tray_id": "blumat_trial",
        "polygon": [[10, 10], [40, 10], [30, 50]],
        "position_index": 0,
    })

    assert cell["region"] == [10, 10, 40, 50]
    assert cell["polygon"] == [[10, 10], [40, 10], [30, 50]]
    assert tray_store.delete_cell("blumat_trial", cell["cell_id"]) == 1
    assert tray_store.list_trays()[0]["cells"] == []
