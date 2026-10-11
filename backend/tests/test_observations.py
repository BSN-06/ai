"""
Tests for observation creation, validation, and retrieval.
"""
import pytest
from datetime import datetime


class TestObservationValidation:
    """Test observation schema validation."""

    def test_create_valid_satellite_observation(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "satellite",
            "observation_timestamp": "2024-06-15T10:00:00",
            "is_utc": True,
            "object_class": "vehicle",
            "confidence": 0.85,
            "bbox_x1": 100.0, "bbox_y1": 200.0, "bbox_x2": 150.0, "bbox_y2": 250.0,
            "latitude": 34.0522, "longitude": -118.2437,
            "coordinate_crs": "WGS84",
            "model_name": "yolov8n",
            "data_provenance": "real_inference",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "id" in data

    def test_create_valid_thermal_observation(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "thermal",
            "observation_timestamp": "2024-06-15T10:02:00",
            "object_class": "person",
            "confidence": 0.72,
            "bbox_x1": 300.0, "bbox_y1": 400.0, "bbox_x2": 340.0, "bbox_y2": 480.0,
            "data_provenance": "real_inference",
        })
        assert response.status_code == 200

    def test_create_valid_radar_observation(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "radar",
            "observation_timestamp": "2024-06-15T10:01:00",
            "radar_track_id": "T-001",
            "position_x": 1000.0, "position_y": 2000.0,
            "velocity_x": 5.0, "velocity_y": -2.0,
            "uncertainty": 10.5,
            "data_provenance": "real_inference",
        })
        assert response.status_code == 200

    def test_invalid_sensor_type_rejected(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "lidar",  # Not allowed
            "data_provenance": "real_inference",
        })
        assert response.status_code == 422

    def test_invalid_confidence_rejected(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "satellite",
            "confidence": 1.5,  # > 1.0
            "data_provenance": "real_inference",
        })
        assert response.status_code == 422

    def test_invalid_latitude_rejected(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "satellite",
            "latitude": 95.0,  # > 90
            "longitude": -118.0,
            "data_provenance": "real_inference",
        })
        assert response.status_code == 422

    def test_invalid_longitude_rejected(self, client):
        response = client.post("/api/observations", json={
            "sensor_type": "thermal",
            "latitude": 34.0,
            "longitude": -200.0,  # < -180
            "data_provenance": "real_inference",
        })
        assert response.status_code == 422

    def test_missing_timestamp_is_allowed(self, client):
        """Observation without timestamp should be accepted (timestamp is optional)."""
        response = client.post("/api/observations", json={
            "sensor_type": "satellite",
            "observation_timestamp": None,
            "object_class": "building",
            "confidence": 0.60,
            "data_provenance": "real_inference",
        })
        assert response.status_code == 200

    def test_list_observations(self, client):
        response = client.get("/api/observations?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "observations" in data

    def test_list_observations_by_sensor(self, client):
        # Create a satellite observation first
        client.post("/api/observations", json={
            "sensor_type": "satellite",
            "data_provenance": "real_inference",
        })
        response = client.get("/api/observations?sensor_type=satellite&limit=5")
        assert response.status_code == 200
        data = response.json()
        for obs in data["observations"]:
            assert obs["sensor_type"] == "satellite"

    def test_observation_pagination(self, client):
        response_1 = client.get("/api/observations?limit=2&offset=0")
        response_2 = client.get("/api/observations?limit=2&offset=2")
        assert response_1.status_code == 200
        assert response_2.status_code == 200
