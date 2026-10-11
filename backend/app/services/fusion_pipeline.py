"""
Fusion pipeline (Module D): correlation -> grouping -> duplicate handling -> prioritization -> events.

Rule-based and explainable; this is NOT a trained fusion model.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.db import models as dbm
from app.services.correlation import correlate_observation_group, check_duplicates, group_correlated
from app.services.observation_normalizer import observation_to_dict
from app.services.prioritization import calculate_priority


def _provenance(obs_dicts: list[dict]) -> str:
    provs = {o.get("data_provenance") for o in obs_dicts}
    if "synthetic" in provs:
        return "synthetic"
    if provs & {"recorded", "recorded_replay", "recorded_prediction"}:
        return "recorded"
    return "real"


def save_decisions(session, decisions: list[dict]) -> None:
    for cr in decisions:
        session.add(dbm.CorrelationDecision(
            id=cr.get("id"),
            observation_ids=cr["observation_ids"], decision_status=cr["decision_status"],
            evidence=cr.get("evidence"), reason=cr.get("reason"), missing_metadata=cr.get("missing_metadata"),
            temporal_delta_s=cr.get("temporal_delta_s"), spatial_distance_m=cr.get("spatial_distance_m"),
            config_params=cr.get("config_params"), event_id=cr.get("event_id"),
        ))


def build_event(obs_dicts: list[dict], decisions: list[dict], duplicate_ids: list[str],
                config: Optional[dict] = None) -> dict:
    cfg = dict(config or {})
    cfg["num_duplicates"] = len(duplicate_ids)
    ev = calculate_priority(obs_dicts, decisions, cfg)
    ev["data_provenance"] = _provenance(obs_dicts)
    if duplicate_ids:
        ev["evidence_summary"] += f" {len(duplicate_ids)} duplicate record(s) linked but not counted."
    return ev


def run_fusion(session, observations: list, config: Optional[dict] = None,
               include_singletons: bool = False, persist: bool = True) -> dict:
    """
    Correlate ORM observations, group them and create one event per group.

    Returns {"decisions": [...], "duplicates": [...], "events": [event dicts]}.
    """
    config = config or {}
    obs_dicts = [observation_to_dict(o) if not isinstance(o, dict) else o for o in observations]
    by_id = {o["id"]: o for o in obs_dicts}
    decisions = correlate_observation_group(obs_dicts, config)
    duplicates = check_duplicates(obs_dicts)
    groups = group_correlated(obs_dicts, decisions, duplicates)

    events = []
    for g in groups:
        if len(g["observation_ids"]) < 2 and not include_singletons:
            continue
        members = [by_id[i] for i in g["observation_ids"]]
        ev = build_event(members, g["decisions"], g["duplicate_ids"], config)
        ev["observation_ids"] = g["observation_ids"] + g["duplicate_ids"]
        ev["duplicate_ids"] = g["duplicate_ids"]
        ev["correlation_ids"] = [d["id"] for d in g["decisions"]]
        for d in g["decisions"]:
            d["event_id"] = ev["id"]
        events.append(ev)

    if persist:
        for ev in events:
            session.add(dbm.Event(
                id=ev["id"], review_priority=ev["review_priority"], source_sensors=ev["source_sensors"],
                observation_ids=ev["observation_ids"], evidence_summary=ev["evidence_summary"],
                uncertainty=ev.get("uncertainty"), missing_information=ev.get("missing_information"),
                priority_explanation=ev["priority_explanation"], data_provenance=ev["data_provenance"],
            ))
        session.flush()
        save_decisions(session, decisions)
        session.commit()
    return {"decisions": decisions, "duplicates": duplicates, "events": events}


def recompute_event(session, event: dbm.Event, config: Optional[dict] = None) -> dbm.Event:
    """Recompute an event's priority after observations were added (used by replay)."""
    obs = session.query(dbm.Observation).filter(dbm.Observation.id.in_(event.observation_ids or [])).all()
    obs_dicts = [observation_to_dict(o) for o in obs]
    decisions = correlate_observation_group(obs_dicts, config)
    duplicates = check_duplicates(obs_dicts)
    dup_ids = list({d["observation_ids"][1] for d in duplicates})
    independent = [o for o in obs_dicts if o["id"] not in dup_ids]
    ev = build_event(independent, decisions, dup_ids, config)
    event.review_priority = ev["review_priority"]
    event.source_sensors = ev["source_sensors"]
    event.evidence_summary = ev["evidence_summary"]
    event.uncertainty = ev.get("uncertainty")
    event.missing_information = ev.get("missing_information")
    event.priority_explanation = ev["priority_explanation"]
    event.updated_at = datetime.utcnow()
    return event
