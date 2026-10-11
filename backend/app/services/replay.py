"""
Recorded-data replay service.

Processes observations in timestamp order with controllable playback.
Clearly labelled as REPLAY — never as a live sensor feed.
"""
import logging
import time
import uuid
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Replays recorded observations in chronological order.
    
    Supports: start, pause, resume, reset, adjustable speed.
    """

    def __init__(self):
        self.session_id: Optional[str] = None
        self.status = "STOPPED"
        self.replay_speed = 1.0
        self.records: list[dict] = []
        self.processed_records: list[dict] = []
        self.current_index = 0
        self.current_timestamp: Optional[datetime] = None
        self.start_timestamp: Optional[datetime] = None
        self.end_timestamp: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._callbacks: list = []
        self.processor = None  # callable(record) -> dict outcome (set by the API layer)
        self.source_file: Optional[str] = None
        self.skipped_records: list[dict] = []
        self.event_timeline: list[dict] = []
        self.errors: list[str] = []
        self.stats = {"saved": 0, "rejected": 0, "events_created": 0, "events_updated": 0}
        self.started_wall: Optional[datetime] = None
        self.completed_wall: Optional[datetime] = None
        self.on_complete = None

    @staticmethod
    def _ts_key(r: dict):
        from app.services.observation_normalizer import parse_timestamp
        return parse_timestamp(r.get("observation_timestamp"))[0]

    def load_records(self, records: list[dict], source_file: Optional[str] = None) -> dict:
        """
        Load records for replay. Records are replayed in recorded-timestamp order; records
        without a valid timestamp cannot be placed on the timeline and are reported as skipped.
        """
        if not records:
            return {"status": "error", "message": "No records provided"}
        if self.status in ("PLAYING", "PAUSED"):
            self.reset()

        valid, skipped = [], []
        for r in records:
            if not isinstance(r, dict):
                skipped.append({"record": str(r)[:80], "reason": "not an object"})
                continue
            try:
                ts = self._ts_key(r)
            except ValueError as e:
                skipped.append({"record_id": r.get("id"), "reason": str(e)})
                continue
            if ts is None:
                skipped.append({"record_id": r.get("id"), "reason": "missing observation_timestamp"})
                continue
            valid.append((ts, r))
        if not valid:
            return {"status": "error", "message": "No records with valid timestamps", "skipped": skipped}
        valid.sort(key=lambda x: x[0])
        sorted_records = [r for _, r in valid]

        self.records = sorted_records
        self.skipped_records = skipped
        self.source_file = source_file
        self.processed_records = []
        self.event_timeline = []
        self.errors = []
        self.stats = {"saved": 0, "rejected": 0, "events_created": 0, "events_updated": 0}
        self.current_index = 0
        self.session_id = str(uuid.uuid4())
        self.status = "LOADED"
        self.current_timestamp = None

        # Extract time range
        self.start_timestamp = valid[0][0]
        self.end_timestamp = valid[-1][0]

        return {
            "status": "loaded",
            "session_id": self.session_id,
            "total_records": len(self.records),
            "skipped_records": len(skipped),
            "skipped": skipped[:20],
            "start_timestamp": str(self.start_timestamp),
            "end_timestamp": str(self.end_timestamp),
            "data_provenance": "recorded_replay",
            "label": "RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED",
        }

    def start(self, callback=None) -> dict:
        """Start replaying records."""
        if not self.records:
            return {"status": "error", "message": "No records loaded"}

        if self.status == "PLAYING":
            return {"status": "already_playing"}

        if self.status == "COMPLETED":
            self.reset()
        self.status = "PLAYING"
        self.started_wall = datetime.utcnow()
        self._stop_event.clear()
        self._pause_event.set()

        if callback:
            self._callbacks = [callback]

        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

        return {
            "status": "playing",
            "session_id": self.session_id,
            "replay_speed": self.replay_speed,
            "label": "RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED",
        }

    def pause(self) -> dict:
        """Pause replay."""
        if self.status != "PLAYING":
            return {"status": "not_playing"}
        self.status = "PAUSED"
        self._pause_event.clear()
        return {"status": "paused", "processed": self.current_index, "total": len(self.records)}

    def resume(self) -> dict:
        """Resume paused replay."""
        if self.status != "PAUSED":
            return {"status": "not_paused"}
        self.status = "PLAYING"
        self._pause_event.set()
        return {"status": "resumed"}

    def reset(self) -> dict:
        """Reset replay to the beginning."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._stop_event.clear()
        self._pause_event.set()
        self.current_index = 0
        self.processed_records = []
        self.event_timeline = []
        self.errors = []
        self.stats = {"saved": 0, "rejected": 0, "events_created": 0, "events_updated": 0}
        self.current_timestamp = None
        self.status = "LOADED" if self.records else "STOPPED"
        return {"status": "reset", "total_records": len(self.records)}

    def set_speed(self, speed: float) -> dict:
        """Set replay speed multiplier."""
        self.replay_speed = max(0.1, min(100.0, speed))
        return {"status": "speed_set", "replay_speed": self.replay_speed}

    def _replay_loop(self):
        """Internal replay loop running in a background thread."""
        while self.current_index < len(self.records) and not self._stop_event.is_set():
            self._pause_event.wait()  # Block if paused
            if self._stop_event.is_set():
                break

            record = self.records[self.current_index]
            ts = self._ts_key(record)

            # Calculate delay based on replay speed
            if self.current_index > 0 and ts:
                prev_ts = self._ts_key(self.records[self.current_index - 1])
                if prev_ts:
                    delta = (ts - prev_ts).total_seconds()
                    if delta > 0 and self.replay_speed > 0:
                        sleep_time = delta / self.replay_speed
                        # Cap sleep to avoid very long waits
                        sleep_time = min(sleep_time, 5.0)
                        if sleep_time > 0.01:
                            self._stop_event.wait(timeout=sleep_time)
                            if self._stop_event.is_set():
                                break

            self.current_timestamp = ts
            # Process the record
            record["_replay_processed_at"] = datetime.utcnow().isoformat()
            record["_replay_label"] = "RECORDED DATA — NOT LIVE INFERENCE"
            if self.processor is not None:
                try:
                    outcome = self.processor(record, self) or {}
                except Exception as e:  # never kill the replay loop
                    outcome = {"status": "ERROR", "error": str(e)}
                    self.errors.append(f"record {record.get('id')}: {e}")
                record["_outcome"] = outcome
            self.processed_records.append(record)

            # Notify callbacks
            for cb in self._callbacks:
                try:
                    cb(record, self.current_index, len(self.records))
                except Exception as e:
                    logger.error(f"Replay callback error: {e}")

            self.current_index += 1

        if self.current_index >= len(self.records):
            self.status = "COMPLETED"
            self.completed_wall = datetime.utcnow()
            if self.on_complete:
                try:
                    self.on_complete(self)
                except Exception as e:
                    self.errors.append(f"completion hook: {e}")
        elif self._stop_event.is_set():
            self.status = "STOPPED"

    def get_status(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "replay_speed": self.replay_speed,
            "total_records": len(self.records),
            "processed_records": len(self.processed_records),
            "current_index": self.current_index,
            "current_timestamp": str(self.current_timestamp) if self.current_timestamp else None,
            "start_timestamp": str(self.start_timestamp) if self.start_timestamp else None,
            "end_timestamp": str(self.end_timestamp) if self.end_timestamp else None,
            "progress": round(self.current_index / len(self.records), 4) if self.records else 0.0,
            "queue_preview": [
                {"id": r.get("id"), "sensor_type": r.get("sensor_type"),
                 "observation_timestamp": str(r.get("observation_timestamp"))}
                for r in self.records[self.current_index:self.current_index + 8]
            ],
            "queue_remaining": max(len(self.records) - self.current_index, 0),
            "skipped_records": len(self.skipped_records),
            "stats": self.stats,
            "event_timeline": self.event_timeline[-50:],
            "errors": self.errors[-20:],
            "source_file": self.source_file,
            "clock": "recorded timestamps (not wall-clock time)",
            "data_provenance": "recorded_replay",
            "label": "RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED",
        }

    def get_processed(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get processed records with pagination."""
        return self.processed_records[offset:offset + limit]


# Singleton instance
replay_engine = ReplayEngine()
