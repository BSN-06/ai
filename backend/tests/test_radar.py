"""
Tests for radar tracking — Kalman filter and data association.
"""
import pytest
import json
import csv
import os
import tempfile
from datetime import datetime, timedelta

from app.services.radar_tracking import KalmanFilter2D, Track, RadarTracker


class TestKalmanFilter2D:

    def test_initialize(self):
        kf = KalmanFilter2D()
        kf.initialize(10.0, 20.0)
        assert abs(kf.position[0] - 10.0) < 1e-6
        assert abs(kf.position[1] - 20.0) < 1e-6

    def test_predict_stationary(self):
        kf = KalmanFilter2D()
        kf.initialize(10.0, 20.0, vx=0.0, vy=0.0)
        predicted = kf.predict(dt=1.0)
        # With zero velocity, prediction stays at same position
        assert abs(predicted[0] - 10.0) < 1.0
        assert abs(predicted[1] - 20.0) < 1.0

    def test_predict_with_velocity(self):
        kf = KalmanFilter2D()
        kf.initialize(0.0, 0.0, vx=5.0, vy=3.0)
        predicted = kf.predict(dt=2.0)
        # Expected: x=10, y=6 after 2 seconds at 5m/s and 3m/s
        assert abs(predicted[0] - 10.0) < 2.0
        assert abs(predicted[1] - 6.0) < 2.0

    def test_update_improves_position(self):
        kf = KalmanFilter2D(measurement_noise=0.1)
        kf.initialize(0.0, 0.0)
        import numpy as np
        updated = kf.update(np.array([5.0, 3.0]))
        # After update toward [5, 3], should move closer
        assert abs(updated[0] - 5.0) < 5.0
        assert abs(updated[1] - 3.0) < 5.0

    def test_uncertainty_increases_with_prediction(self):
        kf = KalmanFilter2D()
        kf.initialize(0.0, 0.0)
        initial_uncertainty = kf.uncertainty
        kf.predict(dt=1.0)
        assert kf.uncertainty >= initial_uncertainty


class TestTrack:

    def test_track_creation(self):
        ts = datetime(2024, 6, 15, 10, 0, 0)
        track = Track("T-001", 100.0, 200.0, ts)
        assert track.track_id == "T-001"
        assert track.status == "ACTIVE"
        assert len(track.history) == 1

    def test_track_update_adds_history(self):
        ts = datetime(2024, 6, 15, 10, 0, 0)
        track = Track("T-001", 100.0, 200.0, ts)
        track.update(105.0, 205.0, ts + timedelta(seconds=1))
        assert len(track.history) == 2

    def test_track_terminates_after_max_misses(self):
        ts = datetime(2024, 6, 15, 10, 0, 0)
        track = Track("T-001", 100.0, 200.0, ts)
        track.max_misses = 3
        for _ in range(3):
            track.mark_missed()
        assert track.status == "TERMINATED"

    def test_track_to_dict_structure(self):
        ts = datetime(2024, 6, 15, 10, 0, 0)
        track = Track("T-001", 100.0, 200.0, ts)
        d = track.to_dict()
        assert "track_id" in d
        assert "status" in d
        assert "current_position" in d
        assert "current_velocity" in d
        assert "history" in d
        assert "uncertainty" in d


class TestRadarTracker:

    def test_process_single_detection(self):
        tracker = RadarTracker()
        ts = datetime(2024, 6, 15, 10, 0, 0)
        detections = [{"timestamp": ts, "x": 100.0, "y": 200.0}]
        result = tracker.process_detections(detections)
        assert result["status"] == "success"
        assert result["total_detections_processed"] == 1
        assert result["active_tracks"] == 1

    def test_process_multiple_detections_creates_tracks(self):
        tracker = RadarTracker()
        ts = datetime(2024, 6, 15, 10, 0, 0)
        detections = [
            {"timestamp": ts, "x": 100.0, "y": 200.0},
            {"timestamp": ts, "x": 500.0, "y": 600.0},  # Far away, new track
        ]
        result = tracker.process_detections(detections)
        assert result["active_tracks"] == 2

    def test_track_association_updates_existing_track(self):
        tracker = RadarTracker(association_threshold=50.0)
        ts = datetime(2024, 6, 15, 10, 0, 0)
        # First detection creates a track
        tracker.process_detections([{"timestamp": ts, "x": 100.0, "y": 200.0}])
        # Second detection nearby should update existing track
        tracker.process_detections([{"timestamp": ts + timedelta(seconds=1), "x": 102.0, "y": 202.0}])
        assert len(tracker.get_tracks()) == 1
        assert tracker.get_tracks()[0]["num_observations"] == 2

    def test_parse_csv_valid(self):
        tracker = RadarTracker()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'x', 'y'])
            writer.writeheader()
            writer.writerow({'timestamp': '2024-06-15T10:00:00', 'x': '100.0', 'y': '200.0'})
            writer.writerow({'timestamp': '2024-06-15T10:00:01', 'x': '105.0', 'y': '205.0'})
            fname = f.name
        try:
            detections = tracker.parse_csv_data(fname)
            assert len(detections) == 2
            assert detections[0]["x"] == 100.0
            assert detections[0]["y"] == 200.0
        finally:
            os.unlink(fname)

    def test_parse_csv_missing_required_column_raises(self):
        tracker = RadarTracker()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'lat'])  # missing 'x' or 'y'
            writer.writeheader()
            writer.writerow({'time': '2024-06-15T10:00:00', 'lat': '34.0'})
            fname = f.name
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                tracker.parse_csv_data(fname)
        finally:
            os.unlink(fname)

    def test_parse_json_valid(self):
        tracker = RadarTracker()
        data = {"detections": [
            {"timestamp": "2024-06-15T10:00:00", "x": 100.0, "y": 200.0},
            {"timestamp": "2024-06-15T10:00:01", "x": 105.0, "y": 205.0},
        ]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            detections = tracker.parse_json_data(fname)
            assert len(detections) == 2
        finally:
            os.unlink(fname)

    def test_reset_clears_tracks(self):
        tracker = RadarTracker()
        ts = datetime(2024, 6, 15, 10, 0, 0)
        tracker.process_detections([{"timestamp": ts, "x": 100.0, "y": 200.0}])
        assert len(tracker.get_tracks()) > 0
        tracker.reset()
        assert len(tracker.get_tracks()) == 0

    def test_empty_detections(self):
        tracker = RadarTracker()
        result = tracker.process_detections([])
        assert result["status"] == "no_data"


class TestRadarAPI:

    def test_radar_import_no_file(self, client):
        response = client.get("/api/observations?sensor_type=radar")
        assert response.status_code == 200

    def test_radar_tracks_endpoint(self, client):
        response = client.get("/api/radar/tracks")
        assert response.status_code == 200
