"""
Tests for observation correlation, duplicate detection, and sensor fusion logic.
"""
import pytest
from datetime import datetime, timedelta

from app.services.correlation import (
    correlate_observations,
    correlate_observation_group,
    check_duplicates,
    haversine_distance_m,
)


class TestHaversineDistance:
    """Test geographic distance calculation."""

    def test_same_point_is_zero(self):
        d = haversine_distance_m(34.0522, -118.2437, 34.0522, -118.2437)
        assert d < 1.0  # Should be essentially 0

    def test_la_to_nyc(self):
        d = haversine_distance_m(34.0522, -118.2437, 40.7128, -74.0060)
        # ~3,940 km
        assert 3_900_000 < d < 4_000_000

    def test_nearby_points(self):
        # ~11m apart
        d = haversine_distance_m(34.0522, -118.2437, 34.0523, -118.2436)
        assert d < 20.0
        assert d > 0.0


class TestObservationCorrelation:
    """Test pairwise correlation logic."""

    def _obs(self, sensor, ts, lat=None, lon=None, crs="WGS84", **kwargs):
        return {
            "id": f"test-{sensor}-{ts}",
            "sensor_type": sensor,
            "observation_timestamp": ts,
            "latitude": lat,
            "longitude": lon,
            "coordinate_crs": crs if lat is not None else None,
            **kwargs,
        }

    def test_fully_compatible_observations_correlated(self):
        """Two observations that are temporally and spatially close should be CORRELATED."""
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=34.0522, lon=-118.2437)
        obs_b = self._obs("radar", base + timedelta(seconds=30), lat=34.0523, lon=-118.2436)
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "CORRELATED"
        assert result["temporal_delta_s"] is not None
        assert result["temporal_delta_s"] < 300
        assert result["spatial_distance_m"] is not None
        assert result["spatial_distance_m"] < 500

    def test_temporally_distant_not_correlated(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=34.0522, lon=-118.2437)
        obs_b = self._obs("thermal", base + timedelta(hours=2), lat=34.0523, lon=-118.2436)
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "NOT_CORRELATED"

    def test_spatially_distant_not_correlated(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=34.0522, lon=-118.2437)
        obs_b = self._obs("radar", base, lat=40.7128, lon=-74.0060)
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "NOT_CORRELATED"

    def test_missing_timestamp_returns_insufficient_metadata(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", None, lat=34.0522, lon=-118.2437)
        obs_b = self._obs("thermal", base, lat=34.0523, lon=-118.2436)
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "INSUFFICIENT_METADATA"
        assert result["missing_metadata"] is not None
        assert any("timestamp" in m for m in result["missing_metadata"])

    def test_missing_coordinates_returns_insufficient_metadata(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=None, lon=None)
        obs_b = self._obs("thermal", base, lat=34.0522, lon=-118.2437)
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "INSUFFICIENT_METADATA"

    def test_pixel_bbox_without_geo_returns_insufficient(self):
        """Image pixel coordinates must not be treated as geographic coordinates."""
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = {"id": "a", "sensor_type": "satellite", "observation_timestamp": base,
                 "bbox_x1": 100.0, "bbox_y1": 200.0, "latitude": None, "longitude": None}
        obs_b = {"id": "b", "sensor_type": "thermal", "observation_timestamp": base,
                 "bbox_x1": 110.0, "bbox_y1": 210.0, "latitude": None, "longitude": None}
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "INSUFFICIENT_METADATA"

    def test_result_has_required_fields(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=34.0522, lon=-118.2437)
        obs_b = self._obs("radar", base, lat=34.0523, lon=-118.2436)
        result = correlate_observations(obs_a, obs_b)
        assert "id" in result
        assert "observation_ids" in result
        assert "decision_status" in result
        assert "evidence" in result
        assert "reason" in result
        assert "config_params" in result

    def test_class_mismatch_noted_in_evidence(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=34.0522, lon=-118.2437, object_class="building")
        obs_b = self._obs("thermal", base, lat=34.0522, lon=-118.2437, object_class="person")
        result = correlate_observations(obs_a, obs_b)
        # Still correlated spatially/temporally, but class disagreement noted
        assert result["decision_status"] == "CORRELATED"
        assert "class_agreement" in result["evidence"]
        assert result["evidence"]["class_agreement"] is False

    def test_custom_temporal_window(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs_a = self._obs("satellite", base, lat=34.0522, lon=-118.2437)
        obs_b = self._obs("radar", base + timedelta(minutes=10), lat=34.0522, lon=-118.2437)
        # With default 300s window (5 min), 10 min gap should NOT correlate
        result = correlate_observations(obs_a, obs_b)
        assert result["decision_status"] == "NOT_CORRELATED"
        # With 700s window, should correlate
        result2 = correlate_observations(obs_a, obs_b, config={"temporal_window_seconds": 700})
        assert result2["decision_status"] == "CORRELATED"


class TestGroupCorrelation:
    """Test correlate_observation_group."""

    def test_group_returns_all_pairs(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        observations = [
            {"id": "a", "sensor_type": "satellite", "observation_timestamp": base,
             "latitude": 34.0, "longitude": -118.0, "coordinate_crs": "WGS84"},
            {"id": "b", "sensor_type": "thermal", "observation_timestamp": base,
             "latitude": 34.0, "longitude": -118.0, "coordinate_crs": "WGS84"},
            {"id": "c", "sensor_type": "radar", "observation_timestamp": base,
             "latitude": 34.0, "longitude": -118.0, "coordinate_crs": "WGS84"},
        ]
        results = correlate_observation_group(observations)
        # 3 obs => C(3,2) = 3 pairs
        assert len(results) == 3

    def test_empty_group_returns_empty(self):
        assert correlate_observation_group([]) == []

    def test_single_observation_returns_empty(self):
        assert correlate_observation_group([{"id": "a"}]) == []


class TestDuplicateDetection:
    """Test duplicate record detection."""

    def test_identical_sensor_close_timestamps_flagged(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs = [
            {"id": "a", "sensor_type": "satellite", "observation_timestamp": base},
            {"id": "b", "sensor_type": "satellite", "observation_timestamp": base + timedelta(seconds=1)},
        ]
        dups = check_duplicates(obs, temporal_threshold_s=5.0)
        assert len(dups) > 0
        assert dups[0]["is_potential_duplicate"] is True

    def test_different_sensors_not_flagged_as_duplicate(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs = [
            {"id": "a", "sensor_type": "satellite", "observation_timestamp": base},
            {"id": "b", "sensor_type": "thermal", "observation_timestamp": base},
        ]
        dups = check_duplicates(obs)
        assert len(dups) == 0

    def test_same_sensor_far_apart_timestamps_not_flagged(self):
        base = datetime(2024, 6, 15, 10, 0, 0)
        obs = [
            {"id": "a", "sensor_type": "satellite", "observation_timestamp": base},
            {"id": "b", "sensor_type": "satellite", "observation_timestamp": base + timedelta(minutes=10)},
        ]
        dups = check_duplicates(obs, temporal_threshold_s=5.0)
        assert len(dups) == 0
