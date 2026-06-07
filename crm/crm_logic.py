from datetime import datetime, timezone

import numpy as np


def segment_duration(value):
    x = float(value)
    if x <= 300:
        return "small"
    if x <= 1200:
        return "normal"
    return "long"


def crm_segment(next_session_segment, weekly_engagement_segment):
    s1 = next_session_segment
    s2 = weekly_engagement_segment

    if s1 == "normal" and s2 == "normal":
        return "stable_normal"
    if s1 == "small" and s2 == "small":
        return "at_risk_short"
    if s1 == "long" and s2 == "long":
        return "high_engaged"
    if s1 == "small" and s2 in ("normal", "long"):
        return "recovering"
    if s1 in ("normal", "long") and s2 == "small":
        return "drop_risk"
    return "mixed"


def crm_risk_score(churn_probability_7d, weekly_mean_7d):
    churn_risk = float(np.clip(churn_probability_7d, 0.0, 1.0))
    weekly_mean = float(max(0.0, weekly_mean_7d))
    activity_risk = 1.0 - float(np.clip(weekly_mean / 1200.0, 0.0, 1.0))
    return 0.6 * churn_risk + 0.4 * activity_risk


def uncertainty_from_crm(churn_probability_7d, weekly_mean_7d):
    score = crm_risk_score(churn_probability_7d, weekly_mean_7d)
    if score >= 0.70:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def short_session_risk(cautious_length_sec):
    x = float(cautious_length_sec)
    if x <= 240:
        return "high"
    if x <= 480:
        return "medium"
    return "low"


def recommended_scenario(short_risk, uncertainty, segment):
    if short_risk == "high":
        return "retention_offer"
    if uncertainty == "high":
        return "light_touch"
    if segment == "high_engaged":
        return "premium_offer"
    return "standard_offer"


def build_crm_payload(
    player_id,
    next_session,
    weekly_engagement,
    model_version="crm_v1",
    generated_at=None,
):
    next_segment = segment_duration(next_session["predicted_length_sec"])
    weekly_segment = segment_duration(weekly_engagement["predicted_mean_session_length_sec"])
    segment = crm_segment(next_segment, weekly_segment)

    churn_probability = float(
        weekly_engagement.get(
            "churn_probability_7d",
            weekly_engagement.get("churn_7d", 0.5),
        )
    )
    weekly_mean = float(weekly_engagement["predicted_mean_session_length_sec"])

    uncertainty = uncertainty_from_crm(churn_probability, weekly_mean)
    short_risk = short_session_risk(next_session["cautious_length_sec"])
    avoid_early_ad = short_risk == "high" or uncertainty == "high"
    scenario = recommended_scenario(short_risk, uncertainty, segment)

    if generated_at is None:
        generated_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    return {
        "player_id": str(player_id),
        "next_session": next_session,
        "weekly_engagement": weekly_engagement,
        "segments": {
            "next_session_segment": next_segment,
            "weekly_engagement_segment": weekly_segment,
            "crm_segment": segment,
        },
        "risk_flags": {
            "short_session_risk": short_risk,
            "prediction_uncertainty": uncertainty,
            "avoid_early_ad": bool(avoid_early_ad),
        },
        "recommended_scenario": scenario,
        "model_metadata": {
            "model_version": model_version,
            "generated_at": generated_at,
        },
    }
