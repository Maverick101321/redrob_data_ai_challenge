"""
reasoning.py — Generate specific, honest, non-hallucinated reasoning per candidate.

Rules (from submission_spec.md Stage 4 evaluation):
  - Never mention skills not actually in the candidate profile
  - Never use identical strings across candidates
  - Never use pure name-insertion templates ("Alice is a good fit because...")
  - Reasoning must be consistent with the rank (rank 1 must sound like rank 1)
  - Keep it 1-2 sentences, specific, data-driven
"""

from __future__ import annotations
from datetime import datetime, date
from typing import Any, Optional

_TODAY = datetime.today().date()


def _days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = date.fromisoformat(str(date_str)[:10])
        return (_TODAY - d).days
    except ValueError:
        return None


# Skills we want to highlight if they appear in the profile
_HIGHLIGHT_SKILLS = [
    "sentence-transformers", "sentence transformers", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "bge", "e5", "embeddings", "vector search", "semantic search",
    "ndcg", "mrr", "map", "learning to rank", "ltr", "ranking",
    "lora", "qlora", "rag", "fine-tuning", "fine tuning",
    "xgboost", "lightgbm", "pytorch", "transformers",
    "python", "kafka", "spark", "redis",
]


def _get_highlight_skills(skills: list[dict]) -> list[str]:
    """Return skills from the candidate's profile that are JD-relevant."""
    found = []
    for s in skills:
        name_lower = s.get("name", "").lower()
        if any(kw in name_lower or name_lower in kw for kw in _HIGHLIGHT_SKILLS):
            prof = s.get("proficiency", "")
            dur = s.get("duration_months", 0) or 0
            if prof and dur > 0:
                found.append(f"{s['name']} ({prof}, {dur}mo)")
            elif prof:
                found.append(f"{s['name']} ({prof})")
            else:
                found.append(s["name"])
    return found[:4]  # top 4 relevant skills


def _get_best_role(career: list) -> Optional[dict]:
    """Return the most recent non-consulting, production-flavoured role."""
    consulting = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    prod_kws = ["production", "deployed", "at scale", "million", "users", "inference"]

    best = None
    best_score = -1
    for role in career:
        company = (role.get("company", "") or "").lower()
        desc = (role.get("description", "") or "").lower()
        is_consulting = any(c in company for c in consulting)
        prod_count = sum(1 for kw in prod_kws if kw in desc)
        dur = role.get("duration_months", 0) or 0
        score = prod_count * 2 + (0 if is_consulting else 3) + min(dur / 12, 3)
        if score > best_score:
            best_score = score
            best = role
    return best


def generate_reasoning(
    candidate: dict[str, Any],
    composite_score: float,
    rank: int,
) -> str:
    """
    Generate a specific, honest 1-2 sentence reasoning string.

    Args:
        candidate: full candidate dict from candidates.jsonl
        composite_score: final score (used to calibrate tone)
        rank: position (1-based)

    Returns:
        str: reasoning text
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])

    yoe = profile.get("years_of_experience", 0) or 0
    title = profile.get("current_title", "Engineer") or "Engineer"
    company = profile.get("current_company", "") or ""
    location = profile.get("location", "") or ""

    # Relevant skills present in profile
    highlight_skills = _get_highlight_skills(skills)

    # Best career role
    best_role = _get_best_role(career)
    best_role_title = best_role.get("title", "") if best_role else ""
    best_role_company = best_role.get("company", "") if best_role else ""
    best_role_dur = (best_role.get("duration_months", 0) or 0) if best_role else 0

    # Behavioral signals
    rrr = signals.get("recruiter_response_rate", None)
    days_inactive = _days_since(signals.get("last_active_date"))
    open_to_work = signals.get("open_to_work_flag", None)
    github = signals.get("github_activity_score", -1)
    notice = signals.get("notice_period_days", None)
    assessment = signals.get("skill_assessment_scores", {}) or {}

    # --- Build sentence 1: career + skills ---
    parts_s1 = []

    if best_role_title and best_role_company:
        if best_role_dur >= 12:
            parts_s1.append(
                f"{yoe:.1f} yrs experience; most recently as {best_role_title} "
                f"at {best_role_company} ({best_role_dur // 12}y {best_role_dur % 12}mo)"
            )
        else:
            parts_s1.append(
                f"{yoe:.1f} yrs experience; currently {title} at {company}"
            )
    else:
        parts_s1.append(f"{yoe:.1f} yrs experience as {title}")

    if highlight_skills:
        parts_s1.append(f"with JD-relevant skills: {', '.join(highlight_skills)}")

    if assessment:
        top_assess = sorted(assessment.items(), key=lambda x: -x[1])[:2]
        assess_str = ", ".join(f"{k} ({v:.0f}/100)" for k, v in top_assess)
        parts_s1.append(f"assessed at {assess_str}")

    sentence1 = "; ".join(parts_s1) + "."

    # --- Build sentence 2: behavioral / availability signals ---
    parts_s2 = []

    if open_to_work is True:
        parts_s2.append("actively open to work")
    elif open_to_work is False:
        parts_s2.append("not currently marked open-to-work")

    if days_inactive is not None:
        if days_inactive <= 7:
            parts_s2.append("active within the past week")
        elif days_inactive <= 30:
            parts_s2.append(f"active {days_inactive}d ago")
        elif days_inactive <= 90:
            parts_s2.append(f"last active ~{days_inactive // 30}mo ago")
        else:
            parts_s2.append(f"inactive for {days_inactive // 30} months")

    if rrr is not None:
        parts_s2.append(f"{rrr:.0%} recruiter response rate")

    if isinstance(github, (int, float)) and github > 40:
        parts_s2.append(f"GitHub activity score {github:.0f}/100")

    if isinstance(notice, (int, float)):
        parts_s2.append(f"{int(notice)}-day notice period")

    if location:
        parts_s2.append(f"based in {location}")

    sentence2 = ("; ".join(parts_s2) + ".") if parts_s2 else ""

    # Combine
    reasoning = (sentence1 + " " + sentence2).strip()

    # Fallback: should never hit this, but just in case
    if len(reasoning) < 20:
        reasoning = (
            f"{yoe:.1f}-year {title} with {len(skills)} skills on profile; "
            f"composite score {composite_score:.3f}."
        )

    return reasoning
