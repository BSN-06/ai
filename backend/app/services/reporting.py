"""
Report persistence: every report is stored in the ``reports`` table and as a JSON file
under ``data/reports``. Report types are kept separate (real evaluation vs replay vs synthetic).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings, BASE_DIR
from app.db import models as dbm

REPORT_DIR = BASE_DIR / settings.REPORT_DIR


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def save_report(session, *, report_type: str, test_mode: str, source: Optional[str],
                model_checkpoint: Optional[str] = None, num_samples: int = 0,
                metrics: Optional[dict] = None, warnings: Optional[list] = None,
                limitations: Optional[list] = None, completion_status: str = "COMPLETED",
                body: Optional[dict] = None) -> dbm.Report:
    rep = dbm.Report(report_type=report_type, test_mode=test_mode, source=source,
                     model_checkpoint=model_checkpoint, num_samples=num_samples, metrics=metrics,
                     warnings=warnings or [], limitations=limitations or [],
                     completion_status=completion_status)
    session.add(rep)
    session.flush()
    if test_mode == "synthetic":
        rep.limitations = list(rep.limitations) + [
            "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA. Verifies software behaviour only; "
            "says nothing about model accuracy or real-world fusion performance."]
    if test_mode == "recorded_replay":
        rep.limitations = list(rep.limitations) + ["RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED."]
    full = to_dict(rep)
    full["body"] = json.loads(json.dumps(body or {}, default=_json_default))
    rep.body = full["body"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{report_type}_{rep.id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=_json_default)
    rep.file_path = str(path)
    session.commit()
    return rep


def to_dict(rep: dbm.Report, include_body: bool = False) -> dict:
    d = {
        "id": rep.id,
        "report_type": rep.report_type,
        "timestamp": (rep.created_at or datetime.utcnow()).isoformat(),
        "test_mode": rep.test_mode,
        "source": rep.source,
        "model_checkpoint": rep.model_checkpoint,
        "num_samples": rep.num_samples,
        "metrics": rep.metrics,
        "warnings": rep.warnings,
        "limitations": rep.limitations,
        "completion_status": rep.completion_status,
    }
    if include_body:
        d["body"] = rep.body
    return d
