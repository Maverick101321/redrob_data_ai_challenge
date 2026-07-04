"""
from __future__ import annotations
honeypot.py — Detect honeypot and fraudulent candidate profiles.

Checks performed:
  1. Timeline inflation: total claimed duration_months >> years_of_experience * 12
  2. Skill impossibility: 'expert' proficiency with 0 duration_months on many skills
  3. Degree timeline impossibility: graduation year conflicts with claimed experience
  4. Mass-expert skills: claims expert on 8+ skills with zero usage duration
"""

from datetime import datetime
from typing import Any, Tuple

_TODAY = datetime.today()

# How many months of overlap/inflation we tolerate (roles can overlap slightly)
_TIMELINE_SLACK_MONTHS = 18
# If expert proficiency + 0 months on this many skills → flag
_EXPERT_ZERO_DURATION_THRESHOLD = 4
# Claimed months vs actual experience months tolerance ratio
_INFLATION_RATIO_THRESHOLD = 1.35


def is_honeypot(candidate: dict) -> Tuple[bool, str]:
    """
    Returns (is_honeypot: bool, reason: str).
    If is_honeypot is True, this candidate must be excluded from the top-100.
    """
    reasons = []

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    years_exp = profile.get("years_of_experience", 0)
    actual_months = years_exp * 12

    # ------------------------------------------------------------------
    # Check 1: Total claimed career duration >> actual experience
    # ------------------------------------------------------------------
    total_career_months = 0
    for role in career:
        dm = role.get("duration_months", 0)
        if isinstance(dm, (int, float)) and dm > 0:
            total_career_months += dm

    if actual_months > 0 and total_career_months > 0:
        ratio = total_career_months / actual_months
        if ratio > _INFLATION_RATIO_THRESHOLD:
            reasons.append(
                f"Career months ({total_career_months:.0f}) inflated vs "
                f"stated experience ({actual_months:.0f} mo); ratio={ratio:.2f}"
            )

    # ------------------------------------------------------------------
    # Check 2: Expert skills with 0 months usage (lazy keyword stuffing)
    # ------------------------------------------------------------------
    expert_zero = [
        s["name"]
        for s in skills
        if str(s.get("proficiency", "")).lower() == "expert"
        and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= _EXPERT_ZERO_DURATION_THRESHOLD:
        reasons.append(
            f"Claims 'expert' on {len(expert_zero)} skills with 0 months duration: "
            f"{', '.join(expert_zero[:5])}"
        )

    # ------------------------------------------------------------------
    # Check 3: Education graduation year conflicts with experience
    # ------------------------------------------------------------------
    for edu in education:
        end_year = edu.get("end_year")
        if end_year and isinstance(end_year, int):
            career_start_implied = _TODAY.year - years_exp
            # Allow 2 year buffer (some people work during studies)
            if end_year > career_start_implied + 2:
                reasons.append(
                    f"Education end year {end_year} is inconsistent with "
                    f"{years_exp:.1f} yrs experience (implied start ~{career_start_implied})"
                )

    # ------------------------------------------------------------------
    # Check 4: Individual role duration impossibly long
    # (role duration > total years_of_experience implies overlap or fabrication)
    # ------------------------------------------------------------------
    for role in career:
        dm = role.get("duration_months", 0)
        title = role.get("title", "role")
        company = role.get("company", "unknown")
        if dm > 0 and actual_months > 0 and dm > actual_months + _TIMELINE_SLACK_MONTHS:
            reasons.append(
                f"Single role '{title}' at '{company}' claims {dm} months "
                f"but total exp is only {actual_months:.0f} months"
            )

    is_flag = len(reasons) > 0
    return is_flag, "; ".join(reasons) if reasons else ""
