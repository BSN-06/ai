"""
Tests for recorded-data replay engine.
"""
import pytest
import time
from datetime import datetime, timedelta

from app.services.replay import ReplayEngine


def _make_records(n=5, base_ts=None):
    if base_ts is None:
        base_ts = datetime(2024, 6, 15, 10, 0, 0)
    return [
        {
            "id": f"rec-{i}",
            "sensor_type": "satellite",
            "observation_timestamp": (base_ts + timedelta(seconds=i * 2)).isoformat(),
            "object_class": "vehicle",
            "confidence": 0.8,
            "data_provenance": "recorded",
        }
        for i in range(n)
    ]


class TestReplayEngine:

    def test_load_records(self):
        engine = ReplayEngine()
        records = _make_records(5)
        result = engine.load_records(records)
        assert result["status"] == "loaded"
        assert result["total_records"] == 5
        assert "REPLAY" in result["label"].upper()
        assert "NOT A LIVE" in result["label"].upper() or "NOT LIVE" in result["label"].upper()

    def test_load_empty_records_returns_error(self):
        engine = ReplayEngine()
        result = engine.load_records([])
        assert result["status"] == "error"

    def test_records_sorted_by_timestamp(self):
        engine = ReplayEngine()
        base = datetime(2024, 6, 15, 10, 0, 0)
        # Provide records out of order
        records = [
            {"observation_timestamp": (base + timedelta(seconds=3)).isoformat()},
            {"observation_timestamp": base.isoformat()},
            {"observation_timestamp": (base + timedelta(seconds=1)).isoformat()},
        ]
        engine.load_records(records)
        # First record should be earliest
        assert engine.records[0]["observation_timestamp"] == base.isoformat()

    def test_start_replay(self):
        engine = ReplayEngine()
        engine.load_records(_make_records(3))
        result = engine.start()
        assert result["status"] == "playing"
        time.sleep(0.1)
        engine._stop_event.set()  # Stop background thread

    def test_pause_and_resume(self):
        engine = ReplayEngine()
        engine.load_records(_make_records(3))
        engine.start()
        pause_result = engine.pause()
        assert pause_result["status"] == "paused"
        resume_result = engine.resume()
        assert resume_result["status"] == "resumed"
        engine._stop_event.set()

    def test_reset(self):
        engine = ReplayEngine()
        engine.load_records(_make_records(3))
        engine.start()
        time.sleep(0.1)
        reset_result = engine.reset()
        assert reset_result["status"] == "reset"
        assert engine.current_index == 0

    def test_set_speed(self):
        engine = ReplayEngine()
        engine.set_speed(2.5)
        assert engine.replay_speed == 2.5

    def test_set_speed_clamps_to_range(self):
        engine = ReplayEngine()
        engine.set_speed(0.001)  # Below minimum
        assert engine.replay_speed >= 0.1
        engine.set_speed(999.0)  # Above maximum
        assert engine.replay_speed <= 100.0

    def test_get_status_has_required_fields(self):
        engine = ReplayEngine()
        engine.load_records(_make_records(3))
        status = engine.get_status()
        assert "status" in status
        assert "replay_speed" in status
        assert "total_records" in status
        assert "processed_records" in status
        assert "label" in status
        assert "data_provenance" in status

    def test_status_label_says_not_live(self):
        engine = ReplayEngine()
        engine.load_records(_make_records(2))
        status = engine.get_status()
        label = status["label"].upper()
        assert "REPLAY" in label or "RECORDED" in label
        assert "NOT" in label and ("LIVE" in label)

    def test_replay_completes_all_records(self):
        engine = ReplayEngine()
        records = _make_records(3)
        engine.load_records(records)
        engine.replay_speed = 100.0  # Fast
        engine.start()
        # Wait for completion
        for _ in range(50):
            if engine.status in ("COMPLETED", "STOPPED"):
                break
            time.sleep(0.1)
        assert engine.status == "COMPLETED"
        assert len(engine.processed_records) == 3

    def test_processed_records_have_replay_label(self):
        engine = ReplayEngine()
        records = _make_records(2)
        engine.load_records(records)
        engine.replay_speed = 1000.0
        engine.start()
        for _ in range(30):
            if engine.status == "COMPLETED":
                break
            time.sleep(0.05)
        for rec in engine.processed_records:
            assert "_replay_label" in rec
            assert "RECORDED" in rec["_replay_label"].upper()

    def test_start_without_loading_returns_error(self):
        engine = ReplayEngine()
        result = engine.start()
        assert result["status"] == "error"


class TestReplayAPI:

    def test_replay_status_endpoint(self, client):
        response = client.get("/api/replay/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_replay_control_reset(self, client):
        response = client.post("/api/replay/control", json={"action": "reset"})
        assert response.status_code == 200
