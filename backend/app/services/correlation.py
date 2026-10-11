"""
Observation correlation and sensor fusion service.

This is a RULE-BASED, uncertainty-aware correlation pipeline.
It is NOT a trained deep-learning fusion model.

Correlation decisions are explainable and include evidence, missing metadata,
and applicable configuration parameters.
"""
import logging
import uuid
import math
from datetime import datetime
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in meters.
    Only valid for geographic coordinates (WGS84 lat/lon).
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def correlate_observations(obs_a: dict, obs_b: dict, config: Optional[dict] = None) -> dict:
    """
    Attempt to correlate two observations from different sensors.
    
    Rules:
    1. Both observations must have timestamps to check temporal compatibility.
    2. Geographic coordinates are required for spatial correlation.
       Pixel bounding boxes are NOT treated as geographic coordinates.
    3. Matching class labels alone are NOT sufficient for correlation.
    4. When metadata is insufficient, returns INSUFFICIENT_METADATA.
    5. Every decision includes evidence and reasoning.
    
    Args:
        obs_a: First observation dict (from DB or schema).
        obs_b: Second observation dict.
        config: Optional configuration overrides.
        
    Returns:
        Correlation decision with full evidence trail.
    """
    if config is None:
        config = {}

    temporal_window = config.get("temporal_window_seconds", settings.TEMPORAL_WINDOW_SECONDS)
    spatial_threshold = config.get("spatial_distance_threshold_m", settings.SPATIAL_DISTANCE_THRESHOLD_M)

    decision_id = str(uuid.uuid4())
    obs_ids = [obs_a.get("id", "unknown_a"), obs_b.get("id", "unknown_b")]
    evidence = {}
    missing_metadata = []
    reasons = []

    # ── Check: Same sensor type is a duplicate check, not correlation ──
    sensor_a = obs_a.get("sensor_type", "unknown")
    sensor_b = obs_b.get("sensor_type", "unknown")
    evidence["sensor_types"] = [sensor_a, sensor_b]

    if sensor_a == sensor_b:
        # Same sensor type — this might be a duplicate, not multi-sensor correlation
        evidence["same_sensor_warning"] = True
        reasons.append("Both observations are from the same sensor type. "
                       "This is a duplicate check, not cross-sensor correlation.")

    # ── Temporal compatibility ──
    ts_a = obs_a.get("observation_timestamp")
    ts_b = obs_b.get("observation_timestamp")

    if ts_a is None:
        missing_metadata.append("observation_a.timestamp")
    if ts_b is None:
        missing_metadata.append("observation_b.timestamp")

    temporal_compatible = False
    temporal_delta_s = None

    if ts_a and ts_b:
        if isinstance(ts_a, str):
            ts_a = datetime.fromisoformat(ts_a)
        if isinstance(ts_b, str):
            ts_b = datetime.fromisoformat(ts_b)

        temporal_delta_s = abs((ts_a - ts_b).total_seconds())
        evidence["temporal_delta_seconds"] = temporal_delta_s
        evidence["temporal_window_seconds"] = temporal_window

        if temporal_delta_s <= temporal_window:
            temporal_compatible = True
            reasons.append(f"Temporal difference ({temporal_delta_s:.1f}s) is within "
                           f"the configured window ({temporal_window}s).")
        else:
            reasons.append(f"Temporal difference ({temporal_delta_s:.1f}s) exceeds "
                           f"the configured window ({temporal_window}s). Not temporally compatible.")

    # ── Spatial compatibility ──
    lat_a = obs_a.get("latitude")
    lon_a = obs_a.get("longitude")
    lat_b = obs_b.get("latitude")
    lon_b = obs_b.get("longitude")

    spatial_compatible = False
    spatial_distance_m = None

    # Check for pixel coordinates being misused as geographic
    has_bbox_a = obs_a.get("bbox_x1") is not None
    has_bbox_b = obs_b.get("bbox_x1") is not None
    has_geo_a = lat_a is not None and lon_a is not None
    has_geo_b = lat_b is not None and lon_b is not None

    if has_bbox_a and not has_geo_a:
        missing_metadata.append("observation_a.geographic_coordinates (has pixel bbox only)")
        evidence["obs_a_has_pixel_bbox_only"] = True

    if has_bbox_b and not has_geo_b:
        missing_metadata.append("observation_b.geographic_coordinates (has pixel bbox only)")
        evidence["obs_b_has_pixel_bbox_only"] = True

    if has_geo_a and has_geo_b:
        # Check CRS compatibility
        crs_a = obs_a.get("coordinate_crs") or "unknown"
        crs_b = obs_b.get("coordinate_crs") or "unknown"

        if crs_a != "unknown" and crs_b != "unknown" and crs_a != crs_b:
            reasons.append(f"Coordinate reference systems differ: {crs_a} vs {crs_b}. "
                           "Spatial comparison may be unreliable without transformation.")
            evidence["crs_mismatch"] = True
        else:
            spatial_distance_m = haversine_distance_m(lat_a, lon_a, lat_b, lon_b)
            evidence["spatial_distance_meters"] = round(spatial_distance_m, 2)
            evidence["spatial_threshold_meters"] = spatial_threshold

            if spatial_distance_m <= spatial_threshold:
                spatial_compatible = True
                reasons.append(f"Spatial distance ({spatial_distance_m:.1f}m) is within "
                               f"the threshold ({spatial_threshold}m).")
            else:
                reasons.append(f"Spatial distance ({spatial_distance_m:.1f}m) exceeds "
                               f"the threshold ({spatial_threshold}m). Not spatially compatible.")
    else:
        if not has_geo_a:
            missing_metadata.append("observation_a.geographic_coordinates")
        if not has_geo_b:
            missing_metadata.append("observation_b.geographic_coordinates")

    # ── Decision logic ──
    if missing_metadata and not (temporal_compatible and spatial_compatible):
        decision_status = "INSUFFICIENT_METADATA"
        reasons.append("Insufficient metadata to make a confident correlation decision. "
                       f"Missing: {', '.join(missing_metadata)}")
    elif temporal_compatible and spatial_compatible:
        decision_status = "CORRELATED"
        reasons.append("Observations are temporally and spatially compatible.")

        # Add confidence/uncertainty information
        conf_a = obs_a.get("confidence")
        conf_b = obs_b.get("confidence")
        if conf_a is not None:
            evidence["confidence_a"] = conf_a
        if conf_b is not None:
            evidence["confidence_b"] = conf_b

        # Check class agreement
        class_a = obs_a.get("object_class")
        class_b = obs_b.get("object_class")
        if class_a and class_b:
            evidence["class_agreement"] = class_a.lower() == class_b.lower()
            if class_a.lower() != class_b.lower():
                reasons.append(f"Note: Object classes differ ({class_a} vs {class_b}). "
                               "Different sensors may use different class taxonomies.")
    elif temporal_compatible and not spatial_compatible:
        decision_status = "NOT_CORRELATED"
        reasons.append("Temporally compatible but spatially too distant.")
    elif spatial_compatible and not temporal_compatible:
        decision_status = "NOT_CORRELATED"
        reasons.append("Spatially compatible but temporally too distant.")
    else:
        decision_status = "NOT_CORRELATED"
        reasons.append("Neither temporally nor spatially compatible.")

    return {
        "id": decision_id,
        "observation_ids": obs_ids,
        "decision_status": decision_status,
        "evidence": evidence,
        "reason": " | ".join(reasons),
        "missing_metadata": missing_metadata if missing_metadata else None,
        "temporal_delta_s": temporal_delta_s,
        "spatial_distance_m": spatial_distance_m,
        "config_params": {
            "temporal_window_seconds": temporal_window,
            "spatial_distance_threshold_m": spatial_threshold,
        },
    }


def correlate_observation_group(observations: list[dict], config: Optional[dict] = None) -> list[dict]:
    """
    Attempt pairwise correlation for a group of observations.
    Returns a list of correlation decisions.
    
    Only cross-sensor pairs are correlated (same-sensor pairs are flagged as duplicates).
    """
    results = []
    seen_pairs = set()

    for i, obs_a in enumerate(observations):
        for j, obs_b in enumerate(observations):
            if i >= j:
                continue
            pair_key = tuple(sorted([obs_a.get("id", str(i)), obs_b.get("id", str(j))]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            results.append(correlate_observations(obs_a, obs_b, config))

    return results


def _as_dt(ts):
    if isinstance(ts, str):
        return datetime.fromisoformat(ts)
    return ts


def check_duplicates(
    observations: list[dict],
    temporal_threshold_s: float = 5.0,
    spatial_threshold_m: float = 25.0,
) -> list[dict]:
    """
    Identify potential duplicate observations (same sensor reporting the same thing twice).

    A pair is a potential duplicate when it comes from the same sensor type AND either
      - shares the same (source_dataset, source_record_id), or
      - is within ``temporal_threshold_s`` AND does not contradict on class AND
        (when both have geographic positions) lies within ``spatial_threshold_m``.
    Each result explains which rule fired.
    """
    duplicates = []
    for i, obs_a in enumerate(observations):
        for j in range(i + 1, len(observations)):
            obs_b = observations[j]
            if obs_a.get("sensor_type") != obs_b.get("sensor_type"):
                continue
            rule = None
            rid_a, rid_b = obs_a.get("source_record_id"), obs_b.get("source_record_id")
            if rid_a and rid_a == rid_b and obs_a.get("source_dataset") == obs_b.get("source_dataset"):
                rule = "same source record id"
            delta = None
            ts_a, ts_b = _as_dt(obs_a.get("observation_timestamp")), _as_dt(obs_b.get("observation_timestamp"))
            if rule is None and ts_a and ts_b:
                delta = abs((ts_a - ts_b).total_seconds())
                if delta > temporal_threshold_s:
                    continue
                ca, cb = obs_a.get("object_class"), obs_b.get("object_class")
                if ca and cb and ca.lower() != cb.lower():
                    continue
                if None not in (obs_a.get("latitude"), obs_a.get("longitude"),
                                obs_b.get("latitude"), obs_b.get("longitude")):
                    d = haversine_distance_m(obs_a["latitude"], obs_a["longitude"],
                                             obs_b["latitude"], obs_b["longitude"])
                    if d > spatial_threshold_m:
                        continue
                rule = "same sensor, close in time, compatible class/position"
            if rule:
                duplicates.append({
                    "observation_ids": [obs_a.get("id"), obs_b.get("id")],
                    "sensor_type": obs_a.get("sensor_type"),
                    "temporal_delta_s": delta,
                    "is_potential_duplicate": True,
                    "rule": rule,
                })
    return duplicates


def group_correlated(observations: list[dict], decisions: list[dict], duplicates: list[dict]) -> list[dict]:
    """
    Build event groups from pairwise CORRELATED decisions (connected components).

    - Duplicates are collapsed: the second record of a duplicate pair is attached to the group
      as ``duplicate_ids`` and is NOT counted as independent evidence.
    - Observations that correlate with nothing form a single-observation group.

    Returns [{"observation_ids": [...independent...], "duplicate_ids": [...],
              "decisions": [...decisions inside the group...]}]
    """
    ids = [o.get("id") for o in observations]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dup_of = {}
    for d in duplicates:
        a, b = d["observation_ids"]
        if a in parent and b in parent and b not in dup_of and a not in dup_of:
            dup_of[b] = a

    for dec in decisions:
        if dec.get("decision_status") != "CORRELATED":
            continue
        a, b = dec["observation_ids"]
        a, b = dup_of.get(a, a), dup_of.get(b, b)
        if a in parent and b in parent and a != b:
            parent[find(a)] = find(b)

    groups: dict[str, dict] = {}
    for oid in ids:
        root = find(dup_of.get(oid, oid))
        g = groups.setdefault(root, {"observation_ids": [], "duplicate_ids": [], "decisions": []})
        (g["duplicate_ids"] if oid in dup_of else g["observation_ids"]).append(oid)

    for dec in decisions:
        a = dec["observation_ids"][0]
        if a in parent:
            root = find(dup_of.get(a, a))
            if dec["observation_ids"][1] in groups[root]["observation_ids"] + groups[root]["duplicate_ids"]:
                groups[root]["decisions"].append(dec)
    return list(groups.values())
