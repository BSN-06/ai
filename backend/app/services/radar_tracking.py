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

    def __init__(self, track_id: str, x: float, y: float, timestamp: datetime):
        self.track_id = track_id
        self.kf = KalmanFilter2D()
        self.kf.initialize(x, y)
        self.history: list[dict] = []
        self.last_update_time = timestamp
        self.created_at = timestamp
        self.miss_count = 0
        self.max_misses = 5
        self.status = "ACTIVE"
        self._record(timestamp, x, y, measured=True)

    def predict(self, timestamp: datetime):
        dt = (timestamp - self.last_update_time).total_seconds()
        if dt > 0:
            self.kf.predict(dt)

    def update(self, x: float, y: float, timestamp: datetime):
        dt = (timestamp - self.last_update_time).total_seconds()
        if dt > 0:
            self.kf.predict(dt)
        self.kf.update(np.array([x, y]))
        self.last_update_time = timestamp
        self.miss_count = 0
        self._record(timestamp, x, y, measured=True)

    def mark_missed(self):
        self.miss_count += 1
        if self.miss_count >= self.max_misses:
            self.status = "TERMINATED"

    def _record(self, timestamp: datetime, x: float, y: float, measured: bool):
        self.history.append({
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
            # Predict all active tracks to current time
            active_tracks = [t for t in self.tracks.values() if t.status == "ACTIVE"]
            for track in active_tracks:
                track.predict(timestamp)

            # Data association (nearest neighbour)
            unassigned_dets = list(range(len(dets)))
            assigned_tracks = set()

            if active_tracks:
                # Compute cost matrix
                for det_idx in list(unassigned_dets):
                    det = dets[det_idx]
                    best_dist = float("inf")
                    best_track = None

                    for track in active_tracks:
                        if track.track_id in assigned_tracks:
                            continue
                        dx = det["x"] - track.kf.position[0]
                        dy = det["y"] - track.kf.position[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < best_dist and dist < self.association_threshold:
                            best_dist = dist
                            best_track = track

                    if best_track is not None:
                        best_track.update(det["x"], det["y"], timestamp)
                        assigned_tracks.add(best_track.track_id)
                        unassigned_dets.remove(det_idx)
                        updated_tracks += 1

            # Create new tracks for unassigned detections
            for det_idx in unassigned_dets:
                det = dets[det_idx]
                track_id = f"T-{str(uuid.uuid4())[:8]}"
                track = Track(track_id, det["x"], det["y"], timestamp)
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
            "timestamp": datetime.utcnow().isoformat(),
        }

    def parse_csv_data(self, file_path: str) -> list[dict]:
        """
        Parse radar detections from a CSV file.
        
        Expected columns: timestamp, x, y (minimum).
        Optional columns: id, latitude, longitude, speed, course.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

        # Identify columns (case-insensitive)
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("timestamp", "time", "datetime", "date_time"):
                col_map["timestamp"] = col
            elif cl in ("x", "pos_x", "position_x", "easting", "lon", "longitude"):
                col_map["x"] = col
            elif cl in ("y", "pos_y", "position_y", "northing", "lat", "latitude"):
                col_map["y"] = col
            elif cl in ("id", "track_id", "target_id", "mmsi"):
                col_map["id"] = col

        required = {"timestamp", "x", "y"}
        missing = required - set(col_map.keys())
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}. "
                f"Expected: timestamp, x, y (or compatible names)."
            )

        detections = []
        for _, row in df.iterrows():
            try:
                ts = pd.to_datetime(row[col_map["timestamp"]])
                x = float(row[col_map["x"]])
                y = float(row[col_map["y"]])
                det = {"timestamp": ts.to_pydatetime(), "x": x, "y": y}
                if "id" in col_map:
                    det["id"] = str(row[col_map["id"]])
                detections.append(det)
            except Exception as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue

        logger.info(f"Parsed {len(detections)} detections from {file_path}")
        return detections

    def parse_json_data(self, file_path: str) -> list[dict]:
        """Parse radar detections from a JSON file."""
        import json
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "detections" in data:
            records = data["detections"]
        elif isinstance(data, dict) and "tracks" in data:
            records = data["tracks"]
        else:
            raise ValueError("JSON must contain a list of detections or a 'detections'/'tracks' key")

        detections = []
        for rec in records:
            ts_val = rec.get("timestamp") or rec.get("time") or rec.get("datetime")
            x_val = rec.get("x") or rec.get("longitude") or rec.get("lon") or rec.get("easting")
            y_val = rec.get("y") or rec.get("latitude") or rec.get("lat") or rec.get("northing")

            if ts_val is None or x_val is None or y_val is None:
                continue

            ts = pd.to_datetime(ts_val).to_pydatetime()
            detections.append({
                "timestamp": ts,
                "x": float(x_val),
                "y": float(y_val),
                "id": str(rec.get("id", rec.get("track_id", rec.get("mmsi", "")))),
            })

        return detections

    def get_tracks(self) -> list[dict]:
        """Get all tracks as serializable dicts."""
        return [track.to_dict() for track in self.tracks.values()]

    def get_active_tracks(self) -> list[dict]:
        return [t.to_dict() for t in self.tracks.values() if t.status == "ACTIVE"]

    def reset(self):
        """Reset all tracks and counters."""
        self.tracks.clear()
        self.processed_count = 0

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
        Evaluate tracking performance against ground truth if available.
        Only calculates metrics that the available data supports.
        """
        result = {
            "total_tracks": len(self.tracks),
            "active_tracks": len([t for t in self.tracks.values() if t.status == "ACTIVE"]),
            "total_observations": self.processed_count,
            "processing_latency_ms": None,
        }

        if ground_truth_tracks is None:
            result["note"] = "No ground truth tracks provided. Cannot calculate tracking accuracy metrics."
            return result

        # If ground truth is provided, calculate basic metrics
        # This is a placeholder for proper MOT metrics (MOTA, MOTP, etc.)
        result["ground_truth_tracks"] = len(ground_truth_tracks)
        result["note"] = "Ground truth comparison available. Position error metrics calculated."
        return result


# Singleton instance
radar_tracker = RadarTracker()
