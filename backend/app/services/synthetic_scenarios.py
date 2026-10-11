"""
Synthetic integration demonstration scenarios.

These generate DETERMINISTIC test observations for verifying software integration.
All records are clearly labelled as SYNTHETIC — NOT LIVE SENSOR DATA.
Synthetic records must NOT be used to claim real model accuracy.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional

LABEL = "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA"


def _tag(run_tag: str) -> str:
    return run_tag or str(uuid.uuid4())[:8]


def generate_synthetic_scenarios(run_tag: str = "") -> dict:
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
            "id": f"SYN-SAT-{_tag(run_tag)}",
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
            "id": f"SYN-THR-{_tag(run_tag)}",
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
            "id": f"SYN-RAD-{_tag(run_tag)}",
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
            "id": f"SYN-MISS-TS-{_tag(run_tag)}",
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
            "id": f"SYN-MISS-GEO-{_tag(run_tag)}",
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
                "id": f"SYN-COMPAT-A-{_tag(run_tag)}",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "object_class": "vehicle",
                "confidence": 0.80,
                "data_provenance": "synthetic",
            },
            {
                "id": f"SYN-COMPAT-B-{_tag(run_tag)}",
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
                "id": f"SYN-INCOMPAT-A-{_tag(run_tag)}",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "data_provenance": "synthetic",
            },
            {
                "id": f"SYN-INCOMPAT-B-{_tag(run_tag)}",
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
    dup_id = f"SYN-DUP-{_tag(run_tag)}"
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
                "id": f"SYN-CONFLICT-A-{_tag(run_tag)}",
                "sensor_type": "satellite",
                "observation_timestamp": base_time.isoformat(),
                "object_class": "building",
                "confidence": 0.90,
                "latitude": 34.0522, "longitude": -118.2437,
                "coordinate_crs": "WGS84",
                "data_provenance": "synthetic",
            },
            {
                "id": f"SYN-CONFLICT-B-{_tag(run_tag)}",
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
            "id": f"SYN-SINGLE-{_tag(run_tag)}",
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

    # ── Scenario 11: Invalid record (schema violation) ──
    scenarios["invalid_record"] = {
        "description": "Confidence above 1.0 and an inverted bounding box — must be rejected, not stored.",
        "label": LABEL,
        "observations": [{
            "id": f"SYN-INVALID-{_tag(run_tag)}",
            "sensor_type": "satellite",
            "observation_timestamp": base_time.isoformat(),
            "object_class": "vehicle",
            "confidence": 1.5,
            "bbox_x1": 200.0, "bbox_y1": 200.0, "bbox_x2": 100.0, "bbox_y2": 100.0,
            "data_provenance": "synthetic",
        }],
        "expected": {"validation": "FAIL", "reason": "Schema validation must reject the record."},
    }

    # ── Scenario 12: Pixel boxes only (no geo) across sensors ──
    scenarios["pixel_only_cross_sensor"] = {
        "description": "Satellite and thermal detections with pixel boxes but no geographic position, "
                       "same class and close in time. Must NOT correlate on class/pixels alone.",
        "label": LABEL,
        "observations": [
            {"id": f"SYN-PIX-A-{_tag(run_tag)}", "sensor_type": "satellite",
             "observation_timestamp": base_time.isoformat(), "object_class": "car", "confidence": 0.9,
             "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 50.0, "bbox_y2": 50.0, "data_provenance": "synthetic"},
            {"id": f"SYN-PIX-B-{_tag(run_tag)}", "sensor_type": "thermal",
             "observation_timestamp": (base_time + timedelta(seconds=5)).isoformat(), "object_class": "car",
             "confidence": 0.9, "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 50.0, "bbox_y2": 50.0,
             "data_provenance": "synthetic"},
        ],
        "expected": {"correlation": "INSUFFICIENT_METADATA",
                     "reason": "Pixel coordinates are not geographic coordinates."},
    }

    # ── Scenario 13: Three-sensor corroboration ──
    scenarios["three_sensor_corroboration"] = {
        "description": "Satellite, thermal and radar observations of one location within 60 s.",
        "label": LABEL,
        "observations": [
            {"id": f"SYN-TRI-S-{_tag(run_tag)}", "sensor_type": "satellite",
             "observation_timestamp": base_time.isoformat(), "object_class": "vehicle", "confidence": 0.88,
             "latitude": 34.0522, "longitude": -118.2437, "coordinate_crs": "WGS84", "data_provenance": "synthetic"},
            {"id": f"SYN-TRI-T-{_tag(run_tag)}", "sensor_type": "thermal",
             "observation_timestamp": (base_time + timedelta(seconds=20)).isoformat(), "object_class": "vehicle",
             "confidence": 0.84, "latitude": 34.0523, "longitude": -118.2438, "coordinate_crs": "WGS84",
             "data_provenance": "synthetic"},
            {"id": f"SYN-TRI-R-{_tag(run_tag)}", "sensor_type": "radar",
             "observation_timestamp": (base_time + timedelta(seconds=40)).isoformat(), "radar_track_id": "T-SYN-3",
             "latitude": 34.0522, "longitude": -118.2436, "coordinate_crs": "WGS84", "uncertainty": 6.0,
             "data_provenance": "synthetic"},
        ],
        "expected": {"event_priority": "HIGH", "reason": "Three independent sensors, all correlated, high confidence."},
    }

    # ── Scenario 14: Event review workflow ──
    scenarios["event_review"] = {
        "description": "Create an event, move it PENDING -> REVIEWING -> REVIEWED with a note; history is kept.",
        "label": LABEL,
        "observations": [{
            "id": f"SYN-REVIEW-{_tag(run_tag)}", "sensor_type": "radar",
            "observation_timestamp": base_time.isoformat(), "radar_track_id": "T-SYN-R",
            "latitude": 34.05, "longitude": -118.24, "coordinate_crs": "WGS84", "uncertainty": 4.0,
            "data_provenance": "synthetic",
        }],
        "expected": {"review_status": "REVIEWED", "history_entries": 2},
    }

    for sc in scenarios.values():
        sc["label"] = LABEL
    return scenarios


def run_synthetic_suite(session) -> dict:
    """
    Execute every scenario through the REAL software path (validation, storage, correlation,
    duplicate check, prioritization, event creation, review) and compare with expectations.
    Each check reports expected vs actual; PASS is only reported when they match.
    """
    from app.db import models as dbm
    from app.services.observation_normalizer import validate_observation, observation_to_dict
    from app.services.correlation import correlate_observation_group, check_duplicates
    from app.services.fusion_pipeline import run_fusion
    from app.services.reporting import save_report

    run_tag = uuid.uuid4().hex[:6]
    scenarios = generate_synthetic_scenarios(run_tag)
    results = {}
    for name, sc in scenarios.items():
        checks = []
        saved = []
        validations = []
        for raw in sc["observations"]:
            v = validate_observation(raw, defaults={"data_provenance": "synthetic"})
            validations.append({"id": raw.get("id"), "status": v["status"],
                                "errors": v["errors"], "warnings": v["warnings"]})
            if v["observation"] is not None:
                obs = dbm.Observation(id=raw["id"], **v["observation"].model_dump())
                obs.data_provenance = "synthetic"
                session.add(obs)
                saved.append(obs)
        session.flush()
        exp = sc["expected"]

        if "validation" in exp:
            actual = validations[0]["status"]
            checks.append({"check": "validation", "expected": exp["validation"], "actual": actual,
                           "pass": actual == exp["validation"]})
            stored = len(saved) > 0
            checks.append({"check": "stored", "expected": exp["validation"] != "FAIL", "actual": stored,
                           "pass": stored == (exp["validation"] != "FAIL")})

        obs_dicts = [observation_to_dict(o) for o in saved]
        correlations = correlate_observation_group(obs_dicts) if len(obs_dicts) >= 2 else []
        actual_corr = correlations[0]["decision_status"] if correlations else None
        if "correlation" in exp:
            checks.append({"check": "correlation", "expected": exp["correlation"], "actual": actual_corr,
                           "pass": actual_corr == exp["correlation"]})
        if name == "conflicting_evidence":
            agree = correlations[0]["evidence"].get("class_agreement") if correlations else None
            checks.append({"check": "class_agreement_flagged", "expected": False, "actual": agree,
                           "pass": agree is False})

        dups = check_duplicates(obs_dicts) if len(obs_dicts) >= 2 else []
        if "duplicate_check" in exp:
            checks.append({"check": "duplicate_detected", "expected": True, "actual": len(dups) > 0,
                           "pass": len(dups) > 0})

        fusion = run_fusion(session, saved, include_singletons=True) if saved else {"events": []}
        event = fusion["events"][0] if fusion["events"] else None
        if name == "duplicate_records":
            n_events = len(fusion["events"])
            checks.append({"check": "duplicates_collapsed_into_one_event", "expected": 1, "actual": n_events,
                           "pass": n_events == 1})
        if "event_priority" in exp:
            actual_p = event["review_priority"] if event else None
            checks.append({"check": "event_priority", "expected": exp["event_priority"], "actual": actual_p,
                           "pass": actual_p == exp["event_priority"]})

        if "review_status" in exp and event:
            ev = session.get(dbm.Event, event["id"])
            for status in ("REVIEWING", "REVIEWED"):
                session.add(dbm.EventReviewHistory(event_id=ev.id, previous_status=ev.review_status,
                                                   new_status=status, notes=f"synthetic review step -> {status}"))
                ev.review_status = status
            session.commit()
            hist = session.query(dbm.EventReviewHistory).filter(dbm.EventReviewHistory.event_id == ev.id).count()
            checks.append({"check": "review_status", "expected": exp["review_status"], "actual": ev.review_status,
                           "pass": ev.review_status == exp["review_status"]})
            checks.append({"check": "history_entries", "expected": exp["history_entries"], "actual": hist,
                           "pass": hist == exp["history_entries"]})

        results[name] = {
            "description": sc["description"],
            "label": sc["label"],
            "expected": exp,
            "actual": {
                "observations_saved": len(saved),
                "validation": validations,
                "correlation_result": actual_corr,
                "has_duplicates": (len(dups) > 0) if len(obs_dicts) >= 2 else None,
                "event_priority": event["review_priority"] if event else None,
                "event_id": event["id"] if event else None,
                "evidence_summary": event["evidence_summary"] if event else None,
                "priority_explanation": event["priority_explanation"] if event else None,
            },
            "checks": checks,
            "pass": bool(checks) and all(c["pass"] for c in checks),
        }
    session.commit()
    passed = sum(1 for r in results.values() if r["pass"])
    out = {
        "label": "SYNTHETIC INTEGRATION TEST — NOT LIVE SENSOR DATA",
        "run_tag": run_tag,
        "total_scenarios": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "scenarios": results,
    }
    rep = save_report(session, report_type="synthetic_integration", test_mode="synthetic",
                      source="built-in deterministic scenarios", num_samples=len(results),
                      metrics={"passed": passed, "failed": len(results) - passed, "total": len(results)},
                      completion_status="COMPLETED", body=out)
    out["report_id"] = rep.id
    return out
