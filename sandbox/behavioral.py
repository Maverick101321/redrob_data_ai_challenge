"""
from __future__ import annotations
behavioral.py — Convert Redrob's 23 behavioral signals into a single multiplier.

The multiplier is in range [0.3, 1.0] and is applied multiplicatively to the
composite score so that an inactive candidate with great skills still ranks low.

Signal groups:
  A. Availability  — open_to_work, last_active_date, notice_period_days
  B. Responsiveness — recruiter_response_rate, avg_response_time_hours, interview_completion_rate
  C. Profile quality — profile_completeness_score, verified_email, verified_phone, linkedin_connected
  D. Activity / visibility — github_activity_score, saved_by_recruiters_30d, search_appearance_30d
  E. Market demand — profile_views_received_30d, endorsements_received, connection_count
"""

from datetime import datetime, date
from typing import Any, Optional

_TODAY = datetime.today().date()


def _days_since(date_str: Optional[str]) -> Optional[int]:
    """Return days since a date string (YYYY-MM-DD). None if unparseable."""
    if not date_str:
        return None
    try:
        d = date.fromisoformat(str(date_str)[:10])
        return (_TODAY - d).days
    except ValueError:
        return None


def compute_behavioral_multiplier(signals: dict[str, Any]) -> float:
    """
    Compute a behavioral multiplier in [0.3, 1.0].

    Args:
        signals: the redrob_signals dict from a candidate record.

    Returns:
        float in [0.3, 1.0]
    """
    if not signals:
        return 0.6  # no signal data → moderate penalty

    mult = 1.0

    # ------------------------------------------------------------------
    # A. Availability signals
    # ------------------------------------------------------------------

    # open_to_work_flag: not open → hard cap on multiplier
    open_to_work = signals.get("open_to_work_flag", None)
    if open_to_work is False:
        mult *= 0.65

    # last_active_date: recency matters a lot
    days_inactive = _days_since(signals.get("last_active_date"))
    if days_inactive is not None:
        if days_inactive > 365:
            mult *= 0.50
        elif days_inactive > 180:
            mult *= 0.70
        elif days_inactive > 90:
            mult *= 0.85
        elif days_inactive > 30:
            mult *= 0.95
        # < 30 days: no penalty

    # notice_period_days: > 90 days is a soft negative for employer
    notice = signals.get("notice_period_days", None)
    if isinstance(notice, (int, float)):
        if notice > 90:
            mult *= 0.88
        elif notice > 60:
            mult *= 0.94

    # ------------------------------------------------------------------
    # B. Responsiveness signals
    # ------------------------------------------------------------------

    # recruiter_response_rate [0.0 – 1.0]: low rate = unlikely to respond
    rrr = signals.get("recruiter_response_rate", None)
    if isinstance(rrr, (int, float)) and 0.0 <= rrr <= 1.0:
        # Map 0→0.55, 0.5→0.80, 1.0→1.0
        mult *= 0.55 + 0.45 * rrr

    # interview_completion_rate [0.0 – 1.0]: ghosting interviews is bad
    icr = signals.get("interview_completion_rate", None)
    if isinstance(icr, (int, float)) and 0.0 <= icr <= 1.0:
        if icr < 0.40:
            mult *= 0.80
        elif icr < 0.70:
            mult *= 0.92

    # avg_response_time_hours: very slow response = slight penalty
    art = signals.get("avg_response_time_hours", None)
    if isinstance(art, (int, float)) and art >= 0:
        if art > 72:
            mult *= 0.90
        elif art > 48:
            mult *= 0.95

    # ------------------------------------------------------------------
    # C. Profile quality signals
    # ------------------------------------------------------------------

    profile_complete = signals.get("profile_completeness_score", None)
    if isinstance(profile_complete, (int, float)):
        if profile_complete < 40:
            mult *= 0.85
        elif profile_complete < 60:
            mult *= 0.93

    # Verification signals: minor boosts for trust
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)
    if verified_email and verified_phone:
        mult *= 1.03
    elif not verified_email and not verified_phone:
        mult *= 0.97

    # ------------------------------------------------------------------
    # D. Activity signals
    # ------------------------------------------------------------------

    # github_activity_score: -1 = no GitHub linked (neutral), >50 = boost
    github = signals.get("github_activity_score", -1)
    if isinstance(github, (int, float)):
        if github > 70:
            mult *= 1.05
        elif github > 50:
            mult *= 1.02
        # -1 or 0–10: no change (don't penalise absence of GitHub)

    # saved_by_recruiters_30d: other recruiters are also interested → positive signal
    saved = signals.get("saved_by_recruiters_30d", 0)
    if isinstance(saved, (int, float)) and saved > 0:
        mult *= min(1.0 + 0.01 * saved, 1.08)  # cap boost at 8%

    # offer_acceptance_rate: -1 = no prior offers (neutral); low rate = slight negative
    oar = signals.get("offer_acceptance_rate", -1)
    if isinstance(oar, (int, float)) and 0.0 <= oar <= 1.0:
        if oar < 0.30:
            mult *= 0.92

    # ------------------------------------------------------------------
    # Final clamp to [0.3, 1.0]
    # ------------------------------------------------------------------
    return round(max(0.30, min(1.0, mult)), 4)


def behavioral_summary(signals: dict[str, Any]) -> dict[str, Any]:
    """Return a human-readable dict of key behavioral signals for debugging."""
    days_inactive = _days_since(signals.get("last_active_date"))
    return {
        "open_to_work": signals.get("open_to_work_flag"),
        "days_since_active": days_inactive,
        "response_rate": signals.get("recruiter_response_rate"),
        "interview_completion": signals.get("interview_completion_rate"),
        "notice_days": signals.get("notice_period_days"),
        "github_score": signals.get("github_activity_score"),
        "saved_30d": signals.get("saved_by_recruiters_30d"),
        "profile_complete": signals.get("profile_completeness_score"),
        "multiplier": compute_behavioral_multiplier(signals),
    }
