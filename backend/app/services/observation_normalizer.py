"""
Observation normalization and metadata validation (Module D, stage 1).

Every sensor module produces a dict that is converted here into the common
observation schema (``ObservationCreate``) and validated before it is saved.

Validation outcome per observation:
- PASS     : all metadata needed for correlation is present and consistent.
- WARNING  : the record is valid but lacks metadata (e.g. timestamp or geographic
             position); it is stored, but correlation will report INSUFFICIENT_METADATA.
- FAIL     : the record violates the schema (bad sensor type, out-of-range values,
             inconsistent bbox, ...). It must not be stored.

Fields that a sensor does not provide stay ``None`` — nothing is invented here.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import ValidationError

from app.schemas.schemas import ObservationCreate

GEOGRAPHIC_CRS = {"WGS84", "EPSG:4326"}
OBSERVATION_FIELDS = set(ObservationCreate.model_fields.keys())


def parse_timestamp(value: Any) -> tuple[Optional[datetime], Optional[bool]]:
    """
    Parse a timestamp into a naive-UTC datetime.

    Returns (datetime_or_None, is_utc_or_None). Offset-aware inputs are converted to UTC.
    Naive inputs are returned as-is with is_utc=None (timezone unknown) — the caller decides
    whether the source declares UTC.
    """
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return None, None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                import pandas as pd
                dt = pd.to_datetime(s).to_pydatetime()
            except Exception:
                raise ValueError(f"Unparseable timestamp: {value!r}")
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None), True
    return dt, None


def normalize_observation(raw: dict, defaults: Optional[dict] = None) -> dict:
    """
    Map an arbitrary sensor-module dict onto the common schema field names.
    Unknown keys are moved into ``metadata_json`` so provenance is preserved.
    """
    data = dict(defaults or {})
    extra: dict = {}
    for key, val in raw.items():
        if key in OBSERVATION_FIELDS:
            data[key] = val
        elif key == "id":
            data.setdefault("source_record_id", str(val))
            extra["original_id"] = val
        elif not key.startswith("_"):
            extra[key] = val

    # Treat NaN (from pandas CSV) as missing
    for k, v in list(data.items()):
        if isinstance(v, float) and math.isnan(v):
            data[k] = None

    if "observation_timestamp" in data:
        ts, is_utc = parse_timestamp(data["observation_timestamp"])
        data["observation_timestamp"] = ts
        if is_utc is not None:
            data["is_utc"] = True
        elif ts is not None and "is_utc" not in raw:
            # Naive timestamp with no declaration: we cannot confirm UTC.
            data["is_utc"] = False

    if extra:
        meta = dict(data.get("metadata_json") or {})
        meta.update({k: v for k, v in extra.items() if _jsonable(v)})
        data["metadata_json"] = meta
    return data


def _jsonable(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool, list, dict))


def validate_observation(raw: dict, defaults: Optional[dict] = None) -> dict:
    """
    Normalize and validate one observation.

    Returns {"status": PASS|WARNING|FAIL, "errors": [...], "warnings": [...],
             "observation": ObservationCreate | None}
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = normalize_observation(raw, defaults)
    except ValueError as e:
        return {"status": "FAIL", "errors": [str(e)], "warnings": [], "observation": None}

    try:
        obs = ObservationCreate(**data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            errors.append(f"{loc}: {err.get('msg')}")
        return {"status": "FAIL", "errors": errors, "warnings": [], "observation": None}

    # ── Semantic checks ──
    if obs.observation_timestamp is None:
        warnings.append("Missing observation_timestamp — temporal correlation impossible.")
    elif not obs.is_utc:
        warnings.append("Timestamp timezone not confirmed as UTC.")

    has_lat = obs.latitude is not None
    has_lon = obs.longitude is not None
    if has_lat != has_lon:
        errors.append("Only one of latitude/longitude supplied.")
    elif not has_lat:
        warnings.append("Missing geographic coordinates — spatial correlation impossible.")
    elif obs.coordinate_crs is None:
        warnings.append("Geographic coordinates without coordinate_crs; WGS84 is NOT assumed.")
    elif obs.coordinate_crs.upper() not in GEOGRAPHIC_CRS:
        warnings.append(f"coordinate_crs '{obs.coordinate_crs}' is not a supported geographic CRS.")

    bbox = [obs.bbox_x1, obs.bbox_y1, obs.bbox_x2, obs.bbox_y2]
    if any(v is not None for v in bbox):
        if any(v is None for v in bbox):
            errors.append("Incomplete bounding box (need x1, y1, x2, y2).")
        elif obs.bbox_x2 <= obs.bbox_x1 or obs.bbox_y2 <= obs.bbox_y1:
            errors.append("Invalid bounding box: x2/y2 must exceed x1/y1.")
        elif min(bbox) < 0:
            errors.append("Bounding box has negative pixel coordinates.")

    if obs.sensor_type in ("satellite", "thermal") and obs.data_provenance == "real_inference":
        if obs.model_name is None:
            warnings.append("Image observation without model_name — provenance incomplete.")
    if obs.sensor_type == "radar" and obs.position_x is None and not has_lat:
        warnings.append("Radar observation without position.")

    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings, "observation": None}
    return {
        "status": "WARNING" if warnings else "PASS",
        "errors": [],
        "warnings": warnings,
        "observation": obs,
    }


def observation_to_dict(o) -> dict:
    """Convert an ORM Observation into the dict shape used by correlation/prioritization."""
    return {
        "id": o.id,
        "sensor_type": o.sensor_type,
        "observation_timestamp": o.observation_timestamp,
        "latitude": o.latitude,
        "longitude": o.longitude,
        "coordinate_crs": o.coordinate_crs,
        "object_class": o.object_class,
        "confidence": o.confidence,
        "bbox_x1": o.bbox_x1,
        "uncertainty": o.uncertainty,
        "data_provenance": o.data_provenance,
        "source_dataset": o.source_dataset,
        "source_record_id": o.source_record_id,
        "radar_track_id": o.radar_track_id,
        "image_path": o.image_path,
        "created_at": o.created_at,
    }
