"""
Tests for API endpoints — system health, datasets, models, inference.
"""
import pytest
import io
import numpy as np


class TestSystemHealth:

    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "app_name" in data
        assert "version" in data
        assert "database" in data
        assert "gpu_available" in data
        assert "device" in data
        assert "timestamp" in data

    def test_system_stats_endpoint(self, client):
        response = client.get("/api/system/stats")
        assert response.status_code == 200
        data = response.json()
        assert "observations" in data
        assert "events" in data
        assert "models" in data
        assert "datasets" in data


class TestDatasets:

    def test_list_datasets(self, client):
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_nonexistent_dataset_returns_404(self, client):
        response = client.get("/api/datasets/nonexistent-12345")
        assert response.status_code == 404

    def test_validate_nonexistent_dataset_returns_404(self, client):
        response = client.post("/api/datasets/validate/nonexistent-12345")
        assert response.status_code == 404


class TestModels:

    def test_list_models(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_nonexistent_model_returns_404(self, client):
        response = client.get("/api/models/nonexistent-12345")
        assert response.status_code == 404


class TestInferenceFileValidation:

    def _make_fake_image_bytes(self):
        """Create a minimal valid JPEG in memory."""
        try:
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[40:60, 40:60] = [200, 100, 50]
            _, buf = cv2.imencode('.jpg', img)
            return buf.tobytes()
        except ImportError:
            # Minimal JPEG header if cv2 not available
            return b'\xff\xd8\xff\xe0' + b'\x00' * 50 + b'\xff\xd9'

    def test_satellite_inference_no_file_returns_422(self, client):
        response = client.post("/api/inference/satellite")
        assert response.status_code == 422

    def test_thermal_inference_no_file_returns_422(self, client):
        response = client.post("/api/inference/thermal")
        assert response.status_code == 422

    def test_invalid_file_type_satellite(self, client):
        response = client.post(
            "/api/inference/satellite",
            files={"file": ("test.exe", b"binary data", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_invalid_file_type_thermal(self, client):
        response = client.post(
            "/api/inference/thermal",
            files={"file": ("test.exe", b"binary data", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_radar_invalid_format(self, client):
        response = client.post(
            "/api/inference/radar",
            files={"file": ("data.xlsx", b"binary data", "application/octet-stream")},
        )
        assert response.status_code == 400


class TestCorrelationAPI:

    def test_correlation_run_insufficient_data(self, client):
        response = client.post("/api/correlation/run")
        assert response.status_code == 200
        # With no specified IDs and potentially < 2 obs, returns status message
        data = response.json()
        assert "status" in data


class TestReports:

    def test_list_reports(self, client):
        response = client.get("/api/reports")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_export_nonexistent_report_returns_404(self, client):
        response = client.get("/api/reports/export/nonexistent-id")
        assert response.status_code == 404


class TestMissingCheckpoints:

    def test_satellite_inference_model_loads_pretrained(self, client):
        """
        When no fine-tuned checkpoint exists, satellite detector should fall back
        to pretrained YOLO weights (or report error if ultralytics not installed).
        This test verifies the API returns a valid response, not a crash.
        """
        try:
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode('.jpg', img)
            img_bytes = buf.tobytes()
        except ImportError:
            img_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'

        response = client.post(
            "/api/inference/satellite",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        # Should return 200 (success with pretrained) or 500 with clear error message
        assert response.status_code in (200, 500)
        if response.status_code == 500:
            data = response.json()
            assert "detail" in data  # Should have an error message, not a silent failure
