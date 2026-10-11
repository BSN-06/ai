"""
Tests for event creation, retrieval, and review workflow.
"""
import pytest
from datetime import datetime


def _create_obs(client, sensor="satellite", lat=34.0, lon=-118.0, ts="2024-06-15T10:00:00"):
    resp = client.post("/api/observations", json={
        "sensor_type": sensor,
        "observation_timestamp": ts,
        "latitude": lat,
        "longitude": lon,
        "coordinate_crs": "WGS84",
        "confidence": 0.75,
        "object_class": "vehicle",
        "data_provenance": "real_inference",
    })
    assert resp.status_code == 200
    return resp.json()["id"]


class TestEventCRUD:

    def test_create_event(self, client):
        obs_id = _create_obs(client)
        response = client.post("/api/events", json={
            "review_priority": "MEDIUM",
            "source_sensors": ["satellite"],
            "observation_ids": [obs_id],
            "evidence_summary": "Single satellite observation.",
            "priority_explanation": "Moderate confidence single sensor.",
            "data_provenance": "real",
        })
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["review_priority"] == "MEDIUM"

    def test_list_events(self, client):
        response = client.get("/api/events")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "events" in data

    def test_get_event_by_id(self, client):
        obs_id = _create_obs(client)
        create_resp = client.post("/api/events", json={
            "review_priority": "LOW",
            "source_sensors": ["satellite"],
            "observation_ids": [obs_id],
            "evidence_summary": "Test event.",
            "data_provenance": "real",
        })
        event_id = create_resp.json()["id"]

        response = client.get(f"/api/events/{event_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == event_id
        assert data["review_priority"] == "LOW"

    def test_get_nonexistent_event_returns_404(self, client):
        response = client.get("/api/events/nonexistent-id-12345")
        assert response.status_code == 404

    def test_list_events_filter_by_priority(self, client):
        client.post("/api/events", json={
            "review_priority": "HIGH",
            "source_sensors": ["satellite"],
            "evidence_summary": "High priority test.",
            "data_provenance": "real",
        })
        response = client.get("/api/events?priority=HIGH")
        assert response.status_code == 200
        for event in response.json()["events"]:
            assert event["review_priority"] == "HIGH"

    def test_invalid_priority_rejected(self, client):
        response = client.post("/api/events", json={
            "review_priority": "CRITICAL",  # Not allowed
            "data_provenance": "real",
        })
        assert response.status_code == 422


class TestEventReviewWorkflow:
    """Test the complete review workflow."""

    def test_update_review_status_to_reviewing(self, client):
        create_resp = client.post("/api/events", json={
            "review_priority": "MEDIUM",
            "source_sensors": ["satellite"],
            "evidence_summary": "Review workflow test.",
            "data_provenance": "real",
        })
        event_id = create_resp.json()["id"]

        response = client.patch(f"/api/events/{event_id}/review", json={
            "review_status": "REVIEWING",
            "notes": "Under investigation",
        })
        assert response.status_code == 200
        assert response.json()["review_status"] == "REVIEWING"

    def test_update_review_status_to_reviewed(self, client):
        create_resp = client.post("/api/events", json={
            "review_priority": "LOW",
            "data_provenance": "real",
        })
        event_id = create_resp.json()["id"]
        client.patch(f"/api/events/{event_id}/review", json={"review_status": "REVIEWING"})
        response = client.patch(f"/api/events/{event_id}/review", json={
            "review_status": "REVIEWED",
            "notes": "No further action required.",
        })
        assert response.status_code == 200
        assert response.json()["review_status"] == "REVIEWED"

    def test_dismiss_event(self, client):
        create_resp = client.post("/api/events", json={
            "review_priority": "LOW",
            "data_provenance": "real",
        })
        event_id = create_resp.json()["id"]
        response = client.patch(f"/api/events/{event_id}/review", json={
            "review_status": "DISMISSED",
            "notes": "False detection.",
        })
        assert response.status_code == 200
        assert response.json()["review_status"] == "DISMISSED"

    def test_invalid_review_status_rejected(self, client):
        create_resp = client.post("/api/events", json={
            "review_priority": "LOW",
            "data_provenance": "real",
        })
        event_id = create_resp.json()["id"]
        response = client.patch(f"/api/events/{event_id}/review", json={
            "review_status": "APPROVED",  # Not allowed
        })
        assert response.status_code == 422

    def test_review_nonexistent_event_returns_404(self, client):
        response = client.patch("/api/events/nonexistent-12345/review", json={
            "review_status": "REVIEWED",
        })
        assert response.status_code == 404


class TestEndToEndWorkflow:
    """
    End-to-end: Create observation → run correlation → create event → review → check retrieval.
    """

    def test_full_workflow(self, client):
        # 1. Create observations
        obs_sat_id = _create_obs(client, "satellite", lat=34.0522, lon=-118.2437,
                                  ts="2024-06-15T10:00:00")
        obs_thm_id = _create_obs(client, "thermal", lat=34.0523, lon=-118.2436,
                                  ts="2024-06-15T10:00:30")

        # 2. Observations exist
        obs_resp = client.get("/api/observations")
        assert obs_resp.status_code == 200
        assert obs_resp.json()["total"] >= 2

        # 3. Run correlation
        corr_resp = client.post(
            f"/api/correlation/run?observation_ids={obs_sat_id}&observation_ids={obs_thm_id}"
        )
        assert corr_resp.status_code == 200
        corr_data = corr_resp.json()
        assert "correlations" in corr_data

        # 4. Create event
        event_resp = client.post("/api/events", json={
            "review_priority": "MEDIUM",
            "source_sensors": ["satellite", "thermal"],
            "observation_ids": [obs_sat_id, obs_thm_id],
            "evidence_summary": "Cross-sensor observation with temporal and spatial compatibility.",
            "data_provenance": "real",
        })
        assert event_resp.status_code == 200
        event_id = event_resp.json()["id"]

        # 5. Retrieve event
        get_resp = client.get(f"/api/events/{event_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["review_priority"] == "MEDIUM"

        # 6. Update review status
        review_resp = client.patch(f"/api/events/{event_id}/review", json={
            "review_status": "REVIEWED",
            "notes": "End-to-end test completed.",
        })
        assert review_resp.status_code == 200
        assert review_resp.json()["review_status"] == "REVIEWED"
