"""
Fusion pipeline, replay pipeline, reports, end-to-end workflow and model/checkpoint handling.
"""
import io
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.services.correlation import correlate_observation_group, check_duplicates, group_correlated
from app.services.observation_normalizer import validate_observation


def _o(i, sensor, lat, lon, t, cls=None, conf=None):
    return {"id": i, "sensor_type": sensor, "observation_timestamp": t, "latitude": lat, "longitude": lon,
            "coordinate_crs": "WGS84", "object_class": cls, "confidence": conf, "data_provenance": "synthetic"}


def test_grouping_creates_separate_groups_and_collapses_duplicates():
    t = datetime(2024, 1, 1)
    obs = [
        _o("a", "satellite", 34.0, -118.0, t, "car", 0.9),
        _o("a2", "satellite", 34.0, -118.0, t + timedelta(seconds=1), "car", 0.88),  # duplicate of a
        _o("b", "radar", 34.0001, -118.0, t + timedelta(seconds=30)),
        _o("c", "thermal", 40.0, -74.0, t, "person", 0.7),  # far away -> own group
    ]
    dec = correlate_observation_group(obs)
    dups = check_duplicates(obs)
    assert [d["observation_ids"] for d in dups] == [["a", "a2"]]
    groups = group_correlated(obs, dec, dups)
    by_member = {tuple(sorted(g["observation_ids"])): g for g in groups}
    assert ("a", "b") in by_member
    assert by_member[("a", "b")]["duplicate_ids"] == ["a2"]
    assert ("c",) in by_member


def test_duplicate_requires_matching_class():
    t = datetime(2024, 1, 1)
    obs = [_o("a", "satellite", 34, -118, t, "car"), _o("b", "satellite", 34, -118, t, "truck")]
    assert check_duplicates(obs) == []


def test_validation_statuses():
    ok = validate_observation({"sensor_type": "radar", "observation_timestamp": "2024-01-01T00:00:00Z",
                               "latitude": 1.0, "longitude": 2.0, "coordinate_crs": "WGS84", "position_x": 1})
    assert ok["status"] == "PASS"
    assert ok["observation"].is_utc is True
    warn = validate_observation({"sensor_type": "thermal"})
    assert warn["status"] == "WARNING"
    bad = validate_observation({"sensor_type": "sonar"})
    assert bad["status"] == "FAIL"
    half = validate_observation({"sensor_type": "radar", "latitude": 1.0})
    assert half["status"] == "FAIL"
    naive = validate_observation({"sensor_type": "radar", "observation_timestamp": "2024-01-01T00:00:00"})
    assert any("UTC" in w for w in naive["warnings"])


def test_observation_api_rejects_invalid_bbox(client):
    r = client.post("/api/observations", json={"sensor_type": "satellite", "bbox_x1": 10, "bbox_y1": 10,
                                               "bbox_x2": 5, "bbox_y2": 20})
    assert r.status_code == 422


def test_end_to_end_workflow(client):
    """Create observations -> validate -> correlate -> event -> retrieve -> review -> report."""
    ts = "2025-03-01T12:00:00Z"
    ids = []
    for sensor, lat, cls, conf in (("satellite", 51.5000, "vehicle", 0.9), ("radar", 51.5001, None, None)):
        r = client.post("/api/observations", json={
            "sensor_type": sensor, "observation_timestamp": ts, "latitude": lat, "longitude": -0.12,
            "coordinate_crs": "WGS84", "object_class": cls, "confidence": conf, "position_x": 1.0,
            "data_provenance": "synthetic", "model_name": "test"})
        assert r.status_code == 200, r.text
        assert r.json()["validation"] in ("PASS", "WARNING")
        ids.append(r.json()["id"])
    corr = client.post("/api/correlation/run", json=ids).json()
    assert corr["correlations"][0]["decision_status"] == "CORRELATED"
    gen = client.post("/api/events/auto-generate", json=ids).json()
    assert gen["events_created"] == 1
    eid = gen["events"][0]["event_id"]
    ev = client.get(f"/api/events/{eid}").json()
    assert sorted(ev["observation_ids"]) == sorted(ids)
    assert len(ev["observations"]) == 2 and len(ev["correlations"]) == 1
    assert client.patch(f"/api/events/{eid}/review",
                        json={"review_status": "REVIEWED", "notes": "checked"}).status_code == 200
    ev = client.get(f"/api/events/{eid}").json()
    assert ev["review_status"] == "REVIEWED" and ev["review_history"][-1]["notes"] == "checked"
    exported = client.get(f"/api/events/{eid}/export").json()
    assert "Not an assessment of intent" in exported["export_note"]
    rep = client.post("/api/reports/fusion-tests").json()
    got = client.get(f"/api/reports/export/{rep['id']}").json()
    assert got["report_type"] == "fusion_tests" and got["metrics"]["CORRELATED"] >= 1


def test_synthetic_run_creates_report_and_all_checks_real(client):
    r = client.post("/api/synthetic/run").json()
    assert r["total_scenarios"] == 14
    for sc in r["scenarios"].values():
        assert sc["checks"], "every scenario must contain at least one real check"
        assert sc["pass"] == all(c["pass"] for c in sc["checks"])
    reports = client.get("/api/reports", params={"report_type": "synthetic_integration"}).json()
    assert any(rep["id"] == r["report_id"] for rep in reports)
    assert any("SYNTHETIC" in l for l in reports[0]["limitations"])


def test_replay_processes_chronologically_and_creates_events(client):
    base = datetime(2024, 6, 15, 10, 0, 0)
    recs = [
        {"id": "r3", "sensor_type": "radar", "observation_timestamp": (base + timedelta(seconds=20)).isoformat() + "Z",
         "latitude": 34.0001, "longitude": -118.0, "coordinate_crs": "WGS84", "radar_track_id": "T1"},
        {"id": "r1", "sensor_type": "satellite", "observation_timestamp": base.isoformat() + "Z",
         "latitude": 34.0, "longitude": -118.0, "coordinate_crs": "WGS84", "object_class": "vehicle", "confidence": 0.9},
        {"id": "r2", "sensor_type": "thermal", "observation_timestamp": (base + timedelta(seconds=10)).isoformat() + "Z",
         "latitude": 34.0, "longitude": -118.0001, "coordinate_crs": "WGS84", "object_class": "vehicle", "confidence": 0.8},
        {"id": "nots", "sensor_type": "radar"},
    ]
    up = client.post("/api/replay/load", files={"file": ("rec.json", json.dumps({"observations": recs}), "application/json")})
    assert up.status_code == 200, up.text
    assert up.json()["total_records"] == 3 and up.json()["skipped_records"] == 1
    client.post("/api/replay/control", json={"action": "start", "speed": 100})
    for _ in range(100):
        st = client.get("/api/replay/status").json()
        if st["status"] == "COMPLETED":
            break
        time.sleep(0.05)
    assert st["status"] == "COMPLETED"
    assert "NOT A LIVE" in st["label"]
    processed = client.get("/api/replay/processed").json()["records"]
    assert [p["id"] for p in processed] == ["r1", "r2", "r3"]
    assert st["stats"]["saved"] == 3
    assert st["stats"]["events_created"] == 1 and st["stats"]["events_updated"] == 1
    assert st["event_timeline"][-1]["observation_count"] == 3
    reports = client.get("/api/reports", params={"report_type": "recorded_replay"}).json()
    assert reports and reports[0]["test_mode"] == "recorded_replay"


def test_radar_api_saves_report_and_observations(client):
    rows = ["timestamp,x,y,target_id"]
    for i in range(10):
        rows.append(f"2024-01-01T00:00:{i:02d},{i * 3.0},{i * 2.0},A")
    r = client.post("/api/inference/radar", files={"file": ("r.csv", "\n".join(rows), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["tracks"]) == 1
    assert d["evaluation"]["association_accuracy"] == 1.0
    assert d["observation_ids"]
    assert client.get(f"/api/reports/export/{d['report_id']}").json()["report_type"] == "radar_tracking"


def test_upload_size_and_empty_file_rejected(client):
    r = client.post("/api/inference/radar", files={"file": ("e.csv", b"", "text/csv")})
    assert r.status_code == 400
    r = client.post("/api/inference/satellite", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_file_endpoint_blocks_traversal(client):
    assert client.get("/api/files/uploads/..%2F..%2Fsurveillance.db").status_code in (400, 404)
    assert client.get("/api/files/secret/x.png").status_code == 400


def test_missing_checkpoint_reports_error_not_fallback():
    from app.services.satellite_detector import SatelliteDetector
    det = SatelliteDetector()
    res = det.load_model(checkpoint_path="/nonexistent/best.pt")
    assert res["status"] == "error"
    assert "Checkpoint not found" in res["message"]
    assert det.is_loaded is False


def test_training_rejected_without_validated_dataset(client):
    model = [m for m in client.get("/api/models").json() if m["sensor_type"] == "satellite"][0]
    r = client.post("/api/training/start", json={"model_id": model["id"], "dataset_name": "xView", "epochs": 1})
    assert r.status_code == 400
    assert r.json()["detail"]["problems"]
    r = client.post("/api/training/start", json={"model_id": model["id"], "dataset_name": "xView", "device": "rm -rf"})
    assert r.status_code == 422


def test_evaluation_rejected_without_labels(client):
    model = client.get("/api/models").json()[0]
    r = client.post("/api/evaluation/start", json={"model_id": model["id"], "dataset_name": "xView"})
    assert r.status_code == 400


def test_models_not_marked_trained(client):
    for m in client.get("/api/models").json():
        assert m["trained"] is False and m["status"] == "PRETRAINED"
        assert m["metrics"] is None
