"""
Synthetic integration demonstration scenarios.

These generate DETERMINISTIC test observations for verifying software integration.
All records are clearly labelled as SYNTHETIC — NOT LIVE SENSOR DATA.
Synthetic records must NOT be used to claim real model accuracy.
"""
import uuid
from datetime import datetime, timedelta


def generate_synthetic_scenarios() -> dict:
    """
    Generate a complete set of synthetic test scenarios covering:
    - Valid records from each sensor
    - Missing timestamps
    - Missing coordinates
    - Compatible observations (should correlate)
    - Incompatible observations (should not correlate)
    - Duplicate records
    - Conflicting evidence
    - Missing sensor input
    - Event creation and review
    
    Returns a dict with scenario name -> list of observations and expected outcomes.
    """
    base_time = datetime(2024, 6, 15, 10, 0, 0)  # Fixed reference time
    scenarios = {}

    # ── Scenario 1: Valid satellite observation ──
    scenarios["valid_satellite"] = {
        "description": "A valid satellite observation with complete metadata.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [{
            "id": f"SYN-SAT-{str(uuid.uuid4())[:8]}",
            "sensor_type": "satellite",
            "observation_timestamp": base_time.isoformat(),
            "is_utc": True,
            "source_dataset": "synthetic_test",
            "object_class": "vehicle",
            "confidence": 0.85,
            "bbox_x1": 100.0, "bbox_y1": 200.0, "bbox_x2": 150.0, "bbox_y2": 250.0,
            "latitude": 34.0522, "longitude": -118.2437,
            "coordinate_crs": "WGS84",
            "model_name": "yolov8n_synthetic",
            "data_provenance": "synthetic",
            "processing_status": "PROCESSED",
        }],
        "expected": {"validation": "PASS", "reason": "All required fields present and valid."},
    }

    # ── Scenario 2: Valid thermal observation ──
    scenarios["valid_thermal"] = {
        "description": "A valid thermal observation with complete metadata.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [{
            "id": f"SYN-THR-{str(uuid.uuid4())[:8]}",
            "sensor_type": "thermal",
            "observation_timestamp": (base_time + timedelta(minutes=2)).isoformat(),
            "is_utc": True,
            "source_dataset": "synthetic_test",
            "object_class": "person",
            "confidence": 0.72,
            "bbox_x1": 300.0, "bbox_y1": 400.0, "bbox_x2": 340.0, "bbox_y2": 480.0,
            "latitude": 34.0525, "longitude": -118.2440,
            "coordinate_crs": "WGS84",
            "model_name": "yolov8n_thermal_synthetic",
            "data_provenance": "synthetic",
            "processing_status": "PROCESSED",
        }],
        "expected": {"validation": "PASS", "reason": "All required fields present and valid."},
    }

    # ── Scenario 3: Valid radar observation ──
    scenarios["valid_radar"] = {
        "description": "A valid radar track observation.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [{
            "id": f"SYN-RAD-{str(uuid.uuid4())[:8]}",
            "sensor_type": "radar",
            "observation_timestamp": (base_time + timedelta(minutes=1)).isoformat(),
            "is_utc": True,
            "source_dataset": "synthetic_test",
            "radar_track_id": "T-SYN-001",
            "position_x": 1000.0, "position_y": 2000.0,
            "velocity_x": 5.0, "velocity_y": -2.0,
            "latitude": 34.0520, "longitude": -118.2435,
            "coordinate_crs": "WGS84",
            "uncertainty": 10.5,
            "data_provenance": "synthetic",
            "processing_status": "PROCESSED",
        }],
        "expected": {"validation": "PASS", "reason": "Radar observation with position and velocity."},
    }

    # ── Scenario 4: Missing timestamp ──
    scenarios["missing_timestamp"] = {
        "description": "An observation with no timestamp — should trigger validation warning.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [{
            "id": f"SYN-MISS-TS-{str(uuid.uuid4())[:8]}",
            "sensor_type": "satellite",
            "observation_timestamp": None,
            "object_class": "building",
            "confidence": 0.60,
            "data_provenance": "synthetic",
        }],
        "expected": {"validation": "WARNING", "reason": "Missing observation_timestamp."},
    }

    # ── Scenario 5: Missing coordinates ──
    scenarios["missing_coordinates"] = {
        "description": "An observation with no geographic coordinates.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [{
            "id": f"SYN-MISS-GEO-{str(uuid.uuid4())[:8]}",
            "sensor_type": "thermal",
            "observation_timestamp": (base_time + timedelta(minutes=3)).isoformat(),
            "object_class": "car",
            "confidence": 0.55,
            "bbox_x1": 50.0, "bbox_y1": 60.0, "bbox_x2": 120.0, "bbox_y2": 140.0,
            "latitude": None, "longitude": None,
            "data_provenance": "synthetic",
        }],
        "expected": {
            "validation": "WARNING",
            "reason": "No geographic coordinates — spatial correlation will return INSUFFICIENT_METADATA.",
        },
    }

    # ── Scenario 6: Compatible observations (should correlate) ──
    scenarios["compatible_observations"] = {
        "description": "Two observations from different sensors that are temporally and spatially compatible.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [
            {
                "id": f"SYN-COMPAT-A-{str(uuid.uuid4())[:8]}",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "object_class": "vehicle",
                "confidence": 0.80,
                "data_provenance": "synthetic",
            },
            {
                "id": f"SYN-COMPAT-B-{str(uuid.uuid4())[:8]}",
                "sensor_type": "radar",
                "observation_timestamp": (base_time + timedelta(seconds=30)).isoformat(),
                "latitude": 34.0523, "longitude": -118.2436,
                "coordinate_crs": "WGS84",
                "radar_track_id": "T-SYN-002",
                "uncertainty": 5.0,
                "data_provenance": "synthetic",
            },
        ],
        "expected": {
            "correlation": "CORRELATED",
            "reason": "Temporally (~30s apart) and spatially (~15m apart) compatible.",
        },
    }

    # ── Scenario 7: Incompatible observations ──
    scenarios["incompatible_observations"] = {
        "description": "Two observations too far apart in space and time to correlate.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [
            {
                "id": f"SYN-INCOMPAT-A-{str(uuid.uuid4())[:8]}",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "data_provenance": "synthetic",
            },
            {
                "id": f"SYN-INCOMPAT-B-{str(uuid.uuid4())[:8]}",
                "sensor_type": "thermal",
                "observation_timestamp": (base_time + timedelta(hours=2)).isoformat(),
                "latitude": 40.7128, "longitude": -74.0060,
                "coordinate_crs": "WGS84",
                "data_provenance": "synthetic",
            },
        ],
        "expected": {
            "correlation": "NOT_CORRELATED",
            "reason": "Too far apart temporally (2 hours) and spatially (~3,940 km).",
        },
    }

    # ── Scenario 8: Duplicate records ──
    dup_id = f"SYN-DUP-{str(uuid.uuid4())[:8]}"
    scenarios["duplicate_records"] = {
        "description": "Two nearly identical observations from the same sensor.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [
            {
                "id": dup_id + "-1",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "object_class": "vehicle",
                "confidence": 0.82,
                "latitude": 34.0522, "longitude": -118.2437,
                "data_provenance": "synthetic",
            },
            {
                "id": dup_id + "-2",
                "sensor_type": "satellite",
                "observation_timestamp": (base_time + timedelta(seconds=1)).isoformat(),
                "object_class": "vehicle",
                "confidence": 0.81,
                "latitude": 34.0522, "longitude": -118.2437,
                "data_provenance": "synthetic",
            },
        ],
        "expected": {
            "duplicate_check": "POTENTIAL_DUPLICATE",
            "reason": "Same sensor, ~1s apart, same class, same location.",
        },
    }

    # ── Scenario 9: Conflicting evidence ──
    scenarios["conflicting_evidence"] = {
        "description": "Two sensors report different classes at the same location.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [
            {
                "id": f"SYN-CONFLICT-A-{str(uuid.uuid4())[:8]}",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "object_class": "building",
                "confidence": 0.90,
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "data_provenance": "synthetic",
            },
            {
                "id": f"SYN-CONFLICT-B-{str(uuid.uuid4())[:8]}",
                "sensor_type": "thermal",
                "observation_timestamp": (base_time + timedelta(seconds=10)).isoformat(),
                "object_class": "person",
                "confidence": 0.75,
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "data_provenance": "synthetic",
            },
        ],
        "expected": {
            "correlation": "CORRELATED",
            "note": "Spatially and temporally compatible but classes disagree. "
                    "Different sensors may use different class taxonomies.",
        },
    }

    # ── Scenario 10: Missing sensor input ──
    scenarios["missing_sensor"] = {
        "description": "Only one sensor type has data — event created with limited evidence.",
        "label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
        "observations": [{
            "id": f"SYN-SINGLE-{str(uuid.uuid4())[:8]}",
            "sensor_type": "satellite",
            "observation_timestamp": base_time.isoformat(),
            "object_class": "vehicle",
            "confidence": 0.70,
            "latitude": 34.0522, "longitude": -118.2437,
            "data_provenance": "synthetic",
        }],
        "expected": {
            "event_priority": "LOW",
            "reason": "Single sensor type, limited corroboration.",
        },
    }

    return scenarios
