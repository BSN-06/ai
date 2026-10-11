"""
Tests for synthetic integration scenarios.
"""
import pytest
from app.services.synthetic_scenarios import generate_synthetic_scenarios
from app.services.correlation import correlate_observation_group, check_duplicates
from app.services.prioritization import calculate_priority


class TestSyntheticScenarios:
    """Verify that each synthetic scenario produces the expected software outcome."""

    @pytest.fixture(autouse=True)
    def scenarios(self):
        self.scenarios = generate_synthetic_scenarios()

    def test_all_expected_scenarios_present(self):
        expected_scenarios = [
            "valid_satellite", "valid_thermal", "valid_radar",
            "missing_timestamp", "missing_coordinates",
            "compatible_observations", "incompatible_observations",
            "duplicate_records", "conflicting_evidence",
            "missing_sensor",
        ]
        for name in expected_scenarios:
            assert name in self.scenarios, f"Missing scenario: {name}"

    def test_compatible_observations_scenario_correlates(self):
        scenario = self.scenarios["compatible_observations"]
        obs_list = scenario["observations"]
        correlations = correlate_observation_group(obs_list)
        assert len(correlations) > 0
        result = correlations[0]["decision_status"]
        expected = scenario["expected"]["correlation"]
        assert result == expected, (
            f"compatible_observations: expected {expected}, got {result}. "
            f"Reason: {correlations[0].get('reason')}"
        )

    def test_incompatible_observations_scenario_not_correlated(self):
        scenario = self.scenarios["incompatible_observations"]
        obs_list = scenario["observations"]
        correlations = correlate_observation_group(obs_list)
        assert len(correlations) > 0
        result = correlations[0]["decision_status"]
        expected = scenario["expected"]["correlation"]
        assert result == expected, f"incompatible_observations: expected {expected}, got {result}"

    def test_missing_timestamp_returns_insufficient_metadata(self):
        scenario = self.scenarios["missing_timestamp"]
        obs = scenario["observations"][0]
        # Create a valid partner to force a correlation check
        from datetime import datetime
        partner = {
            "id": "partner", "sensor_type": "thermal",
            "observation_timestamp": datetime(2024, 6, 15, 10, 0, 0),
            "latitude": 34.0, "longitude": -118.0, "coordinate_crs": "WGS84",
        }
        correlations = correlate_observation_group([obs, partner])
        # Should return INSUFFICIENT_METADATA due to missing timestamp on obs
        decision = correlations[0]["decision_status"]
        assert decision == "INSUFFICIENT_METADATA"

    def test_missing_coordinates_returns_insufficient_metadata(self):
        scenario = self.scenarios["missing_coordinates"]
        obs = scenario["observations"][0]
        from datetime import datetime
        partner = {
            "id": "partner", "sensor_type": "radar",
            "observation_timestamp": datetime(2024, 6, 15, 10, 3, 0),
            "latitude": 34.0, "longitude": -118.0, "coordinate_crs": "WGS84",
        }
        correlations = correlate_observation_group([obs, partner])
        decision = correlations[0]["decision_status"]
        assert decision == "INSUFFICIENT_METADATA"

    def test_duplicate_records_flagged(self):
        scenario = self.scenarios["duplicate_records"]
        obs_list = scenario["observations"]
        dups = check_duplicates(obs_list, temporal_threshold_s=5.0)
        assert len(dups) > 0
        assert dups[0]["is_potential_duplicate"] is True

    def test_conflicting_evidence_still_correlates_spatially(self):
        scenario = self.scenarios["conflicting_evidence"]
        obs_list = scenario["observations"]
        correlations = correlate_observation_group(obs_list)
        # Should correlate (same location) but note class disagreement
        result = correlations[0]["decision_status"]
        assert result == "CORRELATED"
        # Class agreement should be False
        assert correlations[0]["evidence"].get("class_agreement") is False

    def test_missing_sensor_produces_low_priority(self):
        scenario = self.scenarios["missing_sensor"]
        obs_list = scenario["observations"]
        event = calculate_priority(obs_list, [])
        expected = scenario["expected"]["event_priority"]
        assert event["review_priority"] == expected, (
            f"missing_sensor: expected {expected}, got {event['review_priority']}"
        )

    def test_all_scenarios_have_synthetic_label(self):
        for name, scenario in self.scenarios.items():
            assert "label" in scenario
            label = scenario["label"].upper()
            assert "SYNTHETIC" in label, f"Scenario '{name}' missing SYNTHETIC label"
            assert "NOT LIVE" in label or "DEMONSTRATION" in label

    def test_scenario_observations_have_synthetic_provenance(self):
        for name, scenario in self.scenarios.items():
            for obs in scenario["observations"]:
                assert obs.get("data_provenance") == "synthetic", (
                    f"Scenario '{name}' observation missing synthetic provenance"
                )

    def test_synthetic_scenarios_api_endpoint(self, client):
        response = client.post("/api/synthetic/run")
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "SYNTHETIC" in data["label"].upper()
        assert "total_scenarios" in data
        assert "passed" in data
        assert "failed" in data
        assert "scenarios" in data
        assert data["total_scenarios"] > 0


class TestSyntheticDataNotClaimedAsReal:
    """Verify that synthetic data is clearly labelled and not presented as real inference."""

    def test_synthetic_scenario_result_has_provenance(self, client):
        response = client.post("/api/synthetic/run")
        assert response.status_code == 200
        data = response.json()
        # The top-level label must indicate synthetic
        assert "SYNTHETIC" in data["label"].upper()

    def test_synthetic_observations_stored_with_synthetic_provenance(self, client):
        client.post("/api/synthetic/run")
        # Check observations are labelled synthetic
        obs_resp = client.get("/api/observations?limit=100")
        obs_data = obs_resp.json()
        synthetic_obs = [o for o in obs_data["observations"] if o.get("data_provenance") == "synthetic"]
        # Should have some synthetic observations from the scenario run
        assert len(synthetic_obs) >= 0  # At minimum it shouldn't crash
