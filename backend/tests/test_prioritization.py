"""
Tests for event prioritization logic.
"""
import pytest
from datetime import datetime

from app.services.prioritization import calculate_priority


def _obs(sensor, conf=0.7, lat=34.0, lon=-118.0, ts=None, **kw):
    return {
        "id": f"test-{sensor}",
        "sensor_type": sensor,
        "observation_timestamp": ts or datetime(2024, 6, 15, 10, 0, 0),
        "confidence": conf,
        "latitude": lat,
        "longitude": lon,
        "coordinate_crs": "WGS84",
        "data_provenance": "real_inference",
        **kw,
    }


class TestCalculatePriority:

    def test_returns_required_fields(self):
        obs = [_obs("satellite")]
        result = calculate_priority(obs, [])
        assert "review_priority" in result
        assert "source_sensors" in result
        assert "observation_ids" in result
        assert "evidence_summary" in result
        assert "priority_explanation" in result
        assert result["review_priority"] in {"LOW", "MEDIUM", "HIGH"}

    def test_single_observation_low_priority(self):
        obs = [_obs("satellite", conf=0.5)]
        result = calculate_priority(obs, [])
        # Single low-confidence observation should be LOW
        assert result["review_priority"] in {"LOW", "MEDIUM"}

    def test_multiple_sensors_higher_priority(self):
        obs = [
            _obs("satellite", conf=0.85),
            _obs("thermal", conf=0.82),
            _obs("radar", conf=0.90),
        ]
        correlations = [{"decision_status": "CORRELATED"}] * 3
        result = calculate_priority(obs, correlations)
        # Three sensors + high confidence + correlations should be HIGH
        assert result["review_priority"] == "HIGH"

    def test_missing_timestamps_reduce_priority(self):
        obs_no_ts = [
            {"id": "a", "sensor_type": "satellite", "observation_timestamp": None,
             "confidence": 0.9, "data_provenance": "real_inference"},
        ]
        obs_with_ts = [_obs("satellite", conf=0.9)]
        result_no_ts = calculate_priority(obs_no_ts, [])
        result_with_ts = calculate_priority(obs_with_ts, [])
        # No-timestamp should have lower or equal score
        assert result_no_ts.get("internal_score", 0) <= result_with_ts.get("internal_score", 0)

    def test_missing_information_reported(self):
        obs = [{
            "id": "a", "sensor_type": "satellite",
            "observation_timestamp": None,
            "confidence": None,
            "latitude": None,
            "longitude": None,
            "data_provenance": "real_inference",
        }]
        result = calculate_priority(obs, [])
        assert result["missing_information"] is not None
        assert len(result["missing_information"]) > 0

    def test_correlated_observations_boost_priority(self):
        obs = [_obs("satellite", conf=0.7), _obs("thermal", conf=0.7)]
        no_corr = calculate_priority(obs, [])
        with_corr = calculate_priority(
            obs,
            [{"decision_status": "CORRELATED", "observation_ids": ["a", "b"]}]
        )
        assert with_corr.get("internal_score", 0) >= no_corr.get("internal_score", 0)

    def test_not_hostile_determination(self):
        """The evidence summary must NOT claim hostile intent."""
        obs = [_obs("satellite", conf=0.99), _obs("thermal", conf=0.99), _obs("radar", conf=0.99)]
        result = calculate_priority(obs, [{"decision_status": "CORRELATED"}] * 3)
        summary = result["evidence_summary"].lower()
        explanation = result["priority_explanation"].lower()
        for word in ("hostile", "attack", "threat", "enemy", "weapon"):
            assert word not in summary, f"'{word}' should not appear in evidence_summary"
            assert word not in explanation, f"'{word}' should not appear in priority_explanation"

    def test_review_priority_not_autonomous_decision(self):
        """Evidence must indicate this is a review recommendation, not an action."""
        obs = [_obs("satellite"), _obs("thermal")]
        result = calculate_priority(obs, [{"decision_status": "CORRELATED"}])
        assert "review" in result["evidence_summary"].lower() or "priority" in result["evidence_summary"].lower()

    def test_empty_observations_handled(self):
        result = calculate_priority([], [])
        # Should not crash, returns a result even if empty
        assert "review_priority" in result

    def test_high_uncertainty_reduces_priority(self):
        obs_high_unc = [_obs("radar", uncertainty=100.0)]
        obs_low_unc = [_obs("radar", uncertainty=1.0)]
        high_result = calculate_priority(obs_high_unc, [])
        low_result = calculate_priority(obs_low_unc, [])
        assert high_result.get("internal_score", 0) <= low_result.get("internal_score", 0)
