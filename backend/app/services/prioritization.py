"""
Event prioritization service.

This assigns a REVIEW PRIORITY (LOW, MEDIUM, HIGH) as a recommendation
for human review. It is NOT a determination of hostile intent or
a prediction of an attack.

Priority is based on explainable factors:
- Data completeness
- Detection confidence
- Correlation quality
- Agreement/disagreement between observations
- Uncertainty
- Independent supporting observations
- Data freshness
"""
import uuid
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_priority(
    observations: list[dict],
    correlation_decisions: list[dict],
    config: Optional[dict] = None,
) -> dict:
    """
    Calculate review priority for a group of correlated observations.
    
    Returns an event dict with priority, explanation, and evidence.
    """
    if config is None:
        config = {}

    event_id = str(uuid.uuid4())
    factors = []
    score = 0.0  # Internal score 0-100, mapped to LOW/MEDIUM/HIGH

    sensor_types = set()
    observation_ids = []
    total_confidence = 0.0
    confidence_count = 0
    has_missing_data = False
    missing_info = []
    uncertainties = []

    # ── Analyse observations ──
    for obs in observations:
        observation_ids.append(obs.get("id", "unknown"))
        sensor_type = obs.get("sensor_type", "unknown")
        sensor_types.add(sensor_type)

        # Confidence
        conf = obs.get("confidence")
        if conf is not None:
            total_confidence += conf
            confidence_count += 1
        else:
            missing_info.append(f"Missing confidence for {sensor_type} observation")

        # Timestamp
        if obs.get("observation_timestamp") is None:
            has_missing_data = True
            missing_info.append(f"Missing timestamp for {sensor_type} observation")

        # Geographic coordinates
        if obs.get("latitude") is None or obs.get("longitude") is None:
            missing_info.append(f"Missing geographic coordinates for {sensor_type} observation")

        # Uncertainty
        unc = obs.get("uncertainty")
        if unc is not None:
            uncertainties.append(unc)

    # ── Factor 1: Number of independent sensor sources ──
    num_sensors = len(sensor_types)
    if num_sensors >= 3:
        score += 30
        factors.append(f"Observations from {num_sensors} different sensor types (HIGH support).")
    elif num_sensors == 2:
        score += 20
        factors.append(f"Observations from {num_sensors} different sensor types (MODERATE support).")
    else:
        score += 5
        factors.append(f"Observations from only {num_sensors} sensor type(s) (LIMITED support).")

    # ── Factor 2: Average detection confidence ──
    if confidence_count > 0:
        avg_confidence = total_confidence / confidence_count
        if avg_confidence >= 0.8:
            score += 25
            factors.append(f"High average detection confidence: {avg_confidence:.2f}")
        elif avg_confidence >= 0.5:
            score += 15
            factors.append(f"Moderate average detection confidence: {avg_confidence:.2f}")
        else:
            score += 5
            factors.append(f"Low average detection confidence: {avg_confidence:.2f}")
    else:
        factors.append("No confidence scores available.")

    # ── Factor 3: Correlation quality ──
    correlated_count = sum(1 for c in correlation_decisions if c.get("decision_status") == "CORRELATED")
    insufficient_count = sum(1 for c in correlation_decisions if c.get("decision_status") == "INSUFFICIENT_METADATA")

    if correlated_count > 0:
        score += 20
        factors.append(f"{correlated_count} successful correlation(s) between observations.")
    if insufficient_count > 0:
        score -= 5
        factors.append(f"{insufficient_count} correlation(s) had insufficient metadata.")

    # ── Factor 4: Class agreement across sensors ──
    classes = [obs.get("object_class") for obs in observations if obs.get("object_class")]
    if len(classes) >= 2:
        unique_classes = set(c.lower() for c in classes)
        if len(unique_classes) == 1:
            score += 10
            factors.append(f"All observations report the same class: {classes[0]}")
        else:
            score += 5
            factors.append(f"Observations report different classes: {unique_classes}. "
                           "Different sensors may use different class taxonomies.")

    # ── Factor 5: Data completeness ──
    if has_missing_data or missing_info:
        score -= 10
        factors.append(f"Incomplete data: {len(missing_info)} missing field(s).")

    # ── Factor 6: Uncertainty ──
    if uncertainties:
        avg_uncertainty = sum(uncertainties) / len(uncertainties)
        if avg_uncertainty > 50:
            score -= 10
            factors.append(f"High average measurement uncertainty: {avg_uncertainty:.1f}")
        elif avg_uncertainty > 20:
            factors.append(f"Moderate measurement uncertainty: {avg_uncertainty:.1f}")
        else:
            score += 5
            factors.append(f"Low measurement uncertainty: {avg_uncertainty:.1f}")

    # ── Factor 7: Number of observations ──
    num_obs = len(observations)
    if num_obs >= 5:
        score += 10
        factors.append(f"{num_obs} supporting observations.")
    elif num_obs >= 3:
        score += 5
        factors.append(f"{num_obs} supporting observations.")

    # ── Factor 8: Data freshness (relative to reference time) ──
    reference_time = config.get("reference_time")
    timestamps = [o.get("observation_timestamp") for o in observations if o.get("observation_timestamp")]
    if reference_time is not None and timestamps:
        parsed = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in timestamps]
        age_s = (reference_time - max(parsed)).total_seconds()
        stale_after = config.get("stale_after_seconds", 3600)
        if age_s > stale_after:
            score -= 10
            factors.append(f"Stale data: newest observation is {age_s / 60:.0f} min older than the reference time.")
        else:
            factors.append(f"Fresh data: newest observation is {max(age_s, 0) / 60:.0f} min old.")

    # ── Factor 9: Duplicates (reported, never counted as support) ──
    num_duplicates = config.get("num_duplicates", 0)
    if num_duplicates:
        factors.append(f"{num_duplicates} duplicate record(s) excluded from supporting evidence.")

    # ── Map score to priority ──
    score = max(0, min(100, score))

    high_threshold = config.get("high_threshold", 60)
    medium_threshold = config.get("medium_threshold", 30)

    if score >= high_threshold:
        review_priority = "HIGH"
    elif score >= medium_threshold:
        review_priority = "MEDIUM"
    else:
        review_priority = "LOW"

    # ── Build event ──
    evidence_summary = (
        f"Event based on {num_obs} observation(s) from {num_sensors} sensor type(s). "
        f"{'Cross-sensor correlation confirmed. ' if correlated_count > 0 else ''}"
        f"Internal priority score: {score:.0f}/100. "
        f"This is a recommendation for human review only; it does not assess intent "
        f"and is not an operational decision."
    )

    return {
        "id": event_id,
        "review_priority": review_priority,
        "source_sensors": list(sensor_types),
        "observation_ids": observation_ids,
        "evidence_summary": evidence_summary,
        "uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else None,
        "missing_information": missing_info if missing_info else None,
        "priority_explanation": " | ".join(factors),
        "review_status": "PENDING",
        "data_provenance": observations[0].get("data_provenance", "real") if observations else "unknown",
        "created_at": datetime.utcnow(),
        "internal_score": score,
        "factors": factors,
    }
