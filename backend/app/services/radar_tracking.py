"""
Radar track processing service with Kalman filter state estimation.

Architecture decision:
- For structured radar detections / track records, a linear Kalman filter is used
  for state estimation (position and velocity).
- A nearest-neighbour data association method is used for multi-target tracking.
- This processes MARITIME radar data (if DLR dataset is used), NOT aircraft radar data.
- The state vector is [x, y, vx, vy] with a constant-velocity motion model.
- Units and coordinate systems depend on the actual dataset format.
- No neural radar model is claimed — this is classical tracking.
"""
import logging
import uuid
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KalmanFilter2D:
    """
    2D Kalman filter for position-velocity state estimation.
    
    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]
    Motion model: Constant velocity
    
    Assumptions:
    - Linear motion between observations
    - Gaussian process and measurement noise
    - These assumptions may not hold for maneuvering targets
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        Args:
            dt: Time step between measurements (seconds).
            process_noise: Process noise standard deviation.
            measurement_noise: Measurement noise standard deviation.
        """
        self.dt = dt
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        q = process_noise ** 2
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=np.float64) * q

        # Measurement noise covariance
        r = measurement_noise ** 2
        self.R = np.eye(2, dtype=np.float64) * r

        # State and covariance
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Initialize state from first measurement."""
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0

    def predict(self, dt: Optional[float] = None):
        """Predict state forward in time."""
        if dt is not None and dt != self.dt:
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)
        else:
            F = self.F

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        """Update state with a new measurement [x, y]."""
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def position(self):
        return self.x[:2]

    @property
    def velocity(self):
        return self.x[2:]

    @property
    def uncertainty(self):
        """Position uncertainty (trace of position covariance)."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


class Track:
    """Represents a single tracked target."""

    def __init__(self, track_id: str, x: float, y: float, timestamp: datetime, det: Optional[dict] = None):
        self.track_id = track_id
        self.kf = KalmanFilter2D()
        self.kf.initialize(x, y)
        self.history: list[dict] = []
        self.last_update_time = timestamp
        self.created_at = timestamp
        self.miss_count = 0
        self.max_misses = 5
        self.status = "ACTIVE"
        self._record(timestamp, x, y, measured=True, det=det)

    def predicted_position(self, timestamp: datetime) -> np.ndarray:
        """Predicted position at ``timestamp`` without mutating filter state."""
        dt = (timestamp - self.last_update_time).total_seconds()
        return self.kf.position + self.kf.velocity * max(dt, 0.0)

    def predict(self, timestamp: datetime):
        """Kept for API compatibility: prediction is applied inside update() using the true dt."""
        return self.predicted_position(timestamp)

    def update(self, x: float, y: float, timestamp: datetime, det: Optional[dict] = None):
        dt = (timestamp - self.last_update_time).total_seconds()
        if dt > 0:
            self.kf.predict(dt)
        self.kf.update(np.array([x, y]))
        if len(self.history) == 1 and dt > 0:
            # Two-point velocity initialisation (constant-velocity assumption)
            first = self.history[0]
            self.kf.x[2] = (x - first["measured_x"]) / dt
            self.kf.x[3] = (y - first["measured_y"]) / dt
        self.last_update_time = timestamp
        self.miss_count = 0
        self._record(timestamp, x, y, measured=True, det=det)

    def mark_missed(self):
        self.miss_count += 1
        if self.miss_count >= self.max_misses:
            self.status = "TERMINATED"

    def _record(self, timestamp: datetime, x: float, y: float, measured: bool, det: Optional[dict] = None):
        det = det or {}
        self.history.append({
            "source_id": det.get("id"),
            "truth_id": det.get("truth_id"),
            "truth_x": det.get("truth_x"),
            "truth_y": det.get("truth_y"),
            "measured_lat": det.get("lat"),
            "measured_lon": det.get("lon"),
            "timestamp": timestamp.isoformat(),
            "measured_x": x if measured else None,
            "measured_y": y if measured else None,
            "estimated_x": float(self.kf.position[0]),
            "estimated_y": float(self.kf.position[1]),
            "velocity_x": float(self.kf.velocity[0]),
            "velocity_y": float(self.kf.velocity[1]),
            "uncertainty": self.kf.uncertainty,
            "measured": measured,
        })

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "num_observations": len(self.history),
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update_time.isoformat(),
            "current_position": {
                "x": float(self.kf.position[0]),
                "y": float(self.kf.position[1]),
            },
            "current_velocity": {
                "x": float(self.kf.velocity[0]),
                "y": float(self.kf.velocity[1]),
            },
            "uncertainty": self.kf.uncertainty,
            "duration_s": (self.last_update_time - self.created_at).total_seconds(),
            "history": self.history,
        }


class RadarTracker:
    """
    Multi-target tracker for structured radar detections.
    
    Uses nearest-neighbour data association and Kalman filter state estimation.
    Domain: Maritime radar (if DLR dataset) — NOT aircraft radar.
    """

    def __init__(self, association_threshold: float = 50.0, max_misses: int = 5):
        """
        Args:
            association_threshold: Maximum distance (in measurement units) to associate
                a detection with an existing track.
            max_misses: Number of consecutive missed updates before a track is terminated.
        """
        self.tracks: dict[str, Track] = {}
        self.association_threshold = association_threshold
        self.max_misses = max_misses
        self.processed_count = 0
        self.coordinate_crs: Optional[str] = None
        self.units: str = "unknown"
        self.coordinate_mode: str = "cartesian"
        self.geo_origin: Optional[tuple[float, float]] = None
        self.last_processing_ms: Optional[float] = None
        self.warnings: list[str] = []

    def process_detections(self, detections: list[dict]) -> dict:
        """
        Process a batch of radar detections.
        
        Expected detection format:
        {
            "timestamp": datetime or ISO string,
            "x": float,
            "y": float,
            "id": optional string  # source record ID
        }
        
        Returns processing summary.
        """
        if not detections:
            return {"status": "no_data", "message": "No detections provided"}

        start_time = time.time()

        # Sort by timestamp
        for d in detections:
            if isinstance(d.get("timestamp"), str):
                d["timestamp"] = datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00"))

        detections.sort(key=lambda d: d["timestamp"])

        # Group by timestamp for batch association
        from itertools import groupby
        grouped = []
        for ts, group in groupby(detections, key=lambda d: d["timestamp"]):
            grouped.append((ts, list(group)))

        new_tracks = 0
        updated_tracks = 0
        terminated_tracks = 0

        for timestamp, dets in grouped:
            active_tracks = [t for t in self.tracks.values() if t.status == "ACTIVE"]

            # Global nearest-neighbour association (Hungarian assignment with distance gate)
            unassigned_dets = set(range(len(dets)))
            assigned_tracks = set()
            if active_tracks:
                cost = np.full((len(dets), len(active_tracks)), 1e9)
                for di, det in enumerate(dets):
                    for ti, track in enumerate(active_tracks):
                        px, py = track.predicted_position(timestamp)
                        d = float(np.hypot(det["x"] - px, det["y"] - py))
                        if d < self.association_threshold:
                            cost[di, ti] = d
                try:
                    from scipy.optimize import linear_sum_assignment
                    rows, cols = linear_sum_assignment(cost)
                    pairs = list(zip(rows, cols))
                except ImportError:  # greedy fallback
                    pairs, used_r, used_c = [], set(), set()
                    for flat in np.argsort(cost, axis=None):
                        r, c = np.unravel_index(flat, cost.shape)
                        if r not in used_r and c not in used_c:
                            pairs.append((r, c)); used_r.add(r); used_c.add(c)
                for di, ti in pairs:
                    if cost[di, ti] >= 1e9:
                        continue
                    det, track = dets[di], active_tracks[ti]
                    track.update(det["x"], det["y"], timestamp, det=det)
                    assigned_tracks.add(track.track_id)
                    unassigned_dets.discard(di)
                    updated_tracks += 1

            # Create new tracks for unassigned detections
            for det_idx in sorted(unassigned_dets):
                det = dets[det_idx]
                track_id = f"T-{len(self.tracks) + 1:04d}-{str(uuid.uuid4())[:4]}"
                track = Track(track_id, det["x"], det["y"], timestamp, det=det)
                track.max_misses = self.max_misses
                self.tracks[track_id] = track
                new_tracks += 1

            # Mark misses for unassigned active tracks
            for track in active_tracks:
                if track.track_id not in assigned_tracks:
                    track.mark_missed()
                    if track.status == "TERMINATED":
                        terminated_tracks += 1

            self.processed_count += len(dets)

        processing_time_ms = (time.time() - start_time) * 1000
        self.last_processing_ms = processing_time_ms

        return {
            "status": "success",
            "total_detections_processed": self.processed_count,
            "batch_size": len(detections),
            "active_tracks": len([t for t in self.tracks.values() if t.status == "ACTIVE"]),
            "terminated_tracks": len([t for t in self.tracks.values() if t.status == "TERMINATED"]),
            "new_tracks_created": new_tracks,
            "tracks_updated": updated_tracks,
            "tracks_terminated": terminated_tracks,
            "processing_time_ms": round(processing_time_ms, 2),
            "coordinate_mode": self.coordinate_mode,
            "units": self.units,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Column aliases (case-insensitive)
    _ALIASES = {
        "timestamp": ("timestamp", "time", "datetime", "date_time", "utc"),
        "x": ("x", "pos_x", "position_x", "easting", "range_x"),
        "y": ("y", "pos_y", "position_y", "northing", "range_y"),
        "lat": ("lat", "latitude"),
        "lon": ("lon", "lng", "longitude"),
        "id": ("id", "record_id", "detection_id"),
        "truth_id": ("truth_id", "target_id", "track_id", "mmsi", "gt_id"),
        "truth_x": ("truth_x", "gt_x", "true_x"),
        "truth_y": ("truth_y", "gt_y", "true_y"),
    }

    def _normalise_records(self, records: list[dict]) -> list[dict]:
        """
        Convert raw records to detections {timestamp, x, y, ...}.

        Cartesian x/y (e.g. metres) are used directly. If only latitude/longitude exist,
        they are projected to a local East-North plane (metres, equirectangular approximation
        around the first point) — degrees are never mixed with metre thresholds.
        """
        if not records:
            return []
        keys = {k.lower().strip(): k for k in records[0].keys()}
        col = {}
        for canon, aliases in self._ALIASES.items():
            for a in aliases:
                if a in keys:
                    col[canon] = keys[a]
                    break
        if "id" not in col and "truth_id" in col and col["truth_id"].lower() == "id":
            col.pop("truth_id")

        has_xy = "x" in col and "y" in col
        has_geo = "lat" in col and "lon" in col
        if "timestamp" not in col or not (has_xy or has_geo):
            raise ValueError(
                f"Missing required columns. Need a timestamp column and either x/y or latitude/longitude. "
                f"Available columns: {list(records[0].keys())}."
            )
        self.coordinate_mode = "cartesian" if has_xy else "geographic_local_enu"
        self.units = "input units (assumed metres)" if has_xy else "metres (local ENU from WGS84)"
        self.coordinate_crs = None if has_xy else "WGS84 -> local ENU"

        detections, skipped = [], 0
        for rec in records:
            try:
                ts = pd.to_datetime(rec[col["timestamp"]])
                if pd.isna(ts):
                    raise ValueError("missing timestamp")
                ts = ts.to_pydatetime()
                if ts.tzinfo is not None:
                    from datetime import timezone
                    ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                det = {"timestamp": ts}
                if has_geo:
                    lat, lon = float(rec[col["lat"]]), float(rec[col["lon"]])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError("lat/lon out of range")
                    det["lat"], det["lon"] = lat, lon
                if has_xy:
                    det["x"], det["y"] = float(rec[col["x"]]), float(rec[col["y"]])
                    if np.isnan(det["x"]) or np.isnan(det["y"]):
                        raise ValueError("NaN position")
                else:
                    if self.geo_origin is None:
                        self.geo_origin = (det["lat"], det["lon"])
                    lat0, lon0 = self.geo_origin
                    det["x"] = np.radians(det["lon"] - lon0) * 6371000.0 * np.cos(np.radians(lat0))
                    det["y"] = np.radians(det["lat"] - lat0) * 6371000.0
                for k in ("id", "truth_id"):
                    if k in col and rec.get(col[k]) is not None and str(rec.get(col[k])) != "nan":
                        det[k] = str(rec[col[k]])
                for k in ("truth_x", "truth_y"):
                    if k in col and rec.get(col[k]) is not None:
                        det[k] = float(rec[col[k]])
                detections.append(det)
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping invalid radar row: {e}")
        if skipped:
            self.warnings.append(f"{skipped} invalid/missing record(s) skipped during parsing.")
        return detections

    def parse_csv_data(self, file_path: str) -> list[dict]:
        """
        Parse radar detections from a CSV file.

        Required: timestamp + (x, y) or (latitude, longitude).
        Optional: id, truth_id/target_id/mmsi (reference identity), truth_x/truth_y (reference position).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")
        detections = self._normalise_records(df.to_dict(orient="records"))
        logger.info(f"Parsed {len(detections)} detections from {file_path}")
        return detections

    def parse_json_data(self, file_path: str) -> list[dict]:
        """Parse radar detections from a JSON list or {"detections": [...]} / {"tracks": [...]}."""
        import json
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "detections" in data:
            records = data["detections"]
        elif isinstance(data, dict) and "tracks" in data:
            records = data["tracks"]
        else:
            raise ValueError("JSON must contain a list of detections or a 'detections'/'tracks' key")
        return self._normalise_records([r for r in records if isinstance(r, dict)])

    def get_tracks(self) -> list[dict]:
        """Get all tracks as serializable dicts."""
        return [track.to_dict() for track in self.tracks.values()]

    def get_active_tracks(self) -> list[dict]:
        return [t.to_dict() for t in self.tracks.values() if t.status == "ACTIVE"]

    def reset(self):
        """Reset all tracks and counters."""
        self.tracks.clear()
        self.processed_count = 0
        self.geo_origin = None
        self.warnings = []
        self.last_processing_ms = None

    def get_status(self) -> dict:
        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len([t for t in self.tracks.values() if t.status == "ACTIVE"]),
            "terminated_tracks": len([t for t in self.tracks.values() if t.status == "TERMINATED"]),
            "total_detections_processed": self.processed_count,
            "association_threshold": self.association_threshold,
            "max_misses": self.max_misses,
            "coordinate_crs": self.coordinate_crs,
            "units": self.units,
            "domain": "maritime_radar",
            "note": "This tracker processes structured radar detections using a constant-velocity "
                    "Kalman filter. It is designed for maritime radar data and does NOT claim "
                    "aircraft tracking capability.",
        }

    def evaluate(self, ground_truth_tracks: Optional[list[dict]] = None) -> dict:
        """
        Evaluate tracking against reference data embedded in the detections.

        Metrics are computed ONLY when the needed reference fields exist:
        - truth_id (reference target identity, e.g. AIS MMSI or annotated target id):
            association accuracy, track purity, fragmentation, continuity, ID switches.
        - truth_x / truth_y (reference position in the same units as x/y):
            position RMSE / mean error of the Kalman estimate.
        Without reference data only descriptive statistics are returned.
        """
        tracks = list(self.tracks.values())
        result = {
            "total_tracks": len(tracks),
            "active_tracks": len([t for t in tracks if t.status == "ACTIVE"]),
            "total_observations": self.processed_count,
            "processing_latency_ms": round(self.last_processing_ms, 2) if self.last_processing_ms else None,
            "coordinate_mode": self.coordinate_mode,
            "units": self.units,
            "metrics_available": [],
            "metrics_unavailable": [],
        }
        records = [(t.track_id, h) for t in tracks for h in t.history]
        if records:
            lengths = [len(t.history) for t in tracks]
            result["mean_track_length"] = round(float(np.mean(lengths)), 2)
            residuals = [np.hypot(h["measured_x"] - h["estimated_x"], h["measured_y"] - h["estimated_y"])
                         for _, h in records if h.get("measured_x") is not None]
            result["mean_measurement_residual"] = round(float(np.mean(residuals)), 4) if residuals else None

        with_truth = [(tid, h) for tid, h in records if h.get("truth_id")]
        if with_truth:
            from collections import Counter, defaultdict
            per_track = defaultdict(list)
            per_truth = defaultdict(list)
            for tid, h in with_truth:
                per_track[tid].append(h["truth_id"])
                per_truth[h["truth_id"]].append((h["timestamp"], tid))
            majority = {tid: Counter(v).most_common(1)[0][0] for tid, v in per_track.items()}
            correct = sum(1 for tid, h in with_truth if majority[tid] == h["truth_id"])
            fragments, continuity, id_switches = {}, {}, 0
            for gid, items in per_truth.items():
                items.sort()
                tids = [t for _, t in items]
                fragments[gid] = len(set(tids))
                continuity[gid] = Counter(tids).most_common(1)[0][1] / len(tids)
                id_switches += sum(1 for a, b in zip(tids, tids[1:]) if a != b)
            result.update({
                "reference_targets": len(per_truth),
                "association_accuracy": round(correct / len(with_truth), 4),
                "mean_track_purity": round(float(np.mean(
                    [Counter(v).most_common(1)[0][1] / len(v) for v in per_track.values()])), 4),
                "fragmentation": round(float(np.mean([f - 1 for f in fragments.values()])), 4),
                "mean_track_continuity": round(float(np.mean(list(continuity.values()))), 4),
                "id_switches": id_switches,
                "evaluated_records": len(with_truth),
            })
            result["metrics_available"] += ["association_accuracy", "track_purity", "fragmentation",
                                             "continuity", "id_switches"]
        else:
            result["metrics_unavailable"].append(
                "association/continuity/fragmentation (no reference target id column such as truth_id/mmsi)")

        with_pos = [h for _, h in records if h.get("truth_x") is not None and h.get("truth_y") is not None]
        if with_pos:
            errs = np.array([np.hypot(h["estimated_x"] - h["truth_x"], h["estimated_y"] - h["truth_y"])
                             for h in with_pos])
            result.update({
                "position_rmse": round(float(np.sqrt(np.mean(errs ** 2))), 4),
                "position_mean_error": round(float(np.mean(errs)), 4),
                "position_evaluated_points": int(len(errs)),
            })
            result["metrics_available"].append("position_error")
        else:
            result["metrics_unavailable"].append("position error (no truth_x/truth_y reference positions)")

        if not result["metrics_available"]:
            result["note"] = ("No reference data in the input. Only descriptive statistics are reported; "
                              "no tracking accuracy is claimed.")
        return result


# Singleton instance
radar_tracker = RadarTracker()
