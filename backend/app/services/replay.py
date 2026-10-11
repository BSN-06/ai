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

    def load_records(self, records: list[dict]) -> dict:
        """
        Load records for replay. Records must have 'observation_timestamp'.
        """
        if not records:
            return {"status": "error", "message": "No records provided"}

        # Sort by timestamp
        sorted_records = sorted(
            records,
            key=lambda r: r.get("observation_timestamp", "")
        )

        self.records = sorted_records
        self.processed_records = []
        self.current_index = 0
        self.session_id = str(uuid.uuid4())
        self.status = "LOADED"

        # Extract time range
        timestamps = [r.get("observation_timestamp") for r in sorted_records if r.get("observation_timestamp")]
        if timestamps:
            self.start_timestamp = datetime.fromisoformat(timestamps[0]) if isinstance(timestamps[0], str) else timestamps[0]
            self.end_timestamp = datetime.fromisoformat(timestamps[-1]) if isinstance(timestamps[-1], str) else timestamps[-1]

        return {
            "status": "loaded",
            "session_id": self.session_id,
            "total_records": len(self.records),
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

        self.status = "PLAYING"
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
        self.current_index = 0
        self.processed_records = []
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
            ts = record.get("observation_timestamp")
            if ts and isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            self.current_timestamp = ts

            # Calculate delay based on replay speed
            if self.current_index > 0 and ts:
                prev_ts = self.records[self.current_index - 1].get("observation_timestamp")
                if prev_ts:
                    if isinstance(prev_ts, str):
                        prev_ts = datetime.fromisoformat(prev_ts)
                    delta = (ts - prev_ts).total_seconds()
                    if delta > 0 and self.replay_speed > 0:
                        sleep_time = delta / self.replay_speed
                        # Cap sleep to avoid very long waits
                        sleep_time = min(sleep_time, 5.0)
                        if sleep_time > 0.01:
                            self._stop_event.wait(timeout=sleep_time)
                            if self._stop_event.is_set():
                                break

            # Process the record
            record["_replay_processed_at"] = datetime.utcnow().isoformat()
            record["_replay_label"] = "RECORDED DATA — NOT LIVE INFERENCE"
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
            "data_provenance": "recorded_replay",
            "label": "RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED",
        }

    def get_processed(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get processed records with pagination."""
        return self.processed_records[offset:offset + limit]


# Singleton instance
replay_engine = ReplayEngine()
