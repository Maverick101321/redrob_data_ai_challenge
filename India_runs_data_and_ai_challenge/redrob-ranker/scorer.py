"""
scorer.py — Multi-signal composite scorer for the Redrob ranking challenge.

Scoring formula:
  composite = (
      0.35 × semantic_score     (JD ↔ profile cosine similarity)
    + 0.25 × skill_score        (weighted must-have + nice-to-have skills)
    + 0.20 × experience_score   (years, career quality, product vs consulting)
    + 0.10 × location_score     (preferred locations from JD)
    + 0.10 × education_score    (tier + field relevance)
  ) × behavioral_multiplier     (0.3–1.0, from behavioral.py)
"""

from __future__ import annotations
import re
from typing import Any

# ---------------------------------------------------------------------------
# JD-derived skill lists (from job_description.docx analysis)
# ---------------------------------------------------------------------------

# Must-have: these are explicitly required in the JD
MUST_HAVE_SKILLS: list[str] = [
    # Embeddings / retrieval
    "sentence-transformers", "sentence transformers", "bge", "e5", "openai embeddings",
    "embeddings", "dense retrieval", "semantic search", "vector search",
    # Vector databases / hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "pgvector",
    # Ranking / evaluation
    "ndcg", "mrr", "map", "learning to rank", "learning-to-rank", "ltr",
    "ranking", "information retrieval", "ir",
    # Python
    "python",
    # ML / AI systems
    "machine learning", "ml", "nlp", "natural language processing",
    "recommendation system", "recommendation systems", "recommender",
    "a/b testing", "ab testing", "offline evaluation",
]

# Nice-to-have: present → bonus, absent → not penalised
NICE_TO_HAVE_SKILLS: list[str] = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "lightgbm", "gradient boosting",
    "rag", "retrieval augmented generation",
    "transformers", "bert", "llm", "large language model",
    "distributed systems", "kafka", "spark",
    "ray", "triton", "onnx", "torchserve",
    "redis", "milvus", "ann",
    "pytorch", "tensorflow",
    "docker", "kubernetes", "mlflow", "kubeflow",
]

# Consulting / IT-services companies that JD explicitly says to down-weight
CONSULTING_COMPANIES: set[str] = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "mindtree", "ltts",
}

# Preferred locations from JD
PREFERRED_LOCATIONS: set[str] = {
    "noida", "pune", "delhi", "ncr", "gurugram", "gurgaon", "faridabad",
    "hyderabad", "mumbai", "bengaluru", "bangalore", "chennai",
}

# Proficiency weights
PROFICIENCY_WEIGHTS: dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.80,
    "intermediate": 0.55,
    "beginner": 0.25,
    "": 0.40,  # unspecified
}

# Education tier weights
EDU_TIER_WEIGHTS: dict[str, float] = {
    "tier_1": 1.0,
    "tier1": 1.0,
    "tier_2": 0.70,
    "tier2": 0.70,
    "tier_3": 0.40,
    "tier3": 0.40,
}

# Relevant fields of study
RELEVANT_FIELDS: list[str] = [
    "computer science", "cs", "ai", "artificial intelligence",
    "machine learning", "data science", "statistics", "mathematics",
    "electronics", "ece", "electrical", "information technology", "it",
    "software engineering",
]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _skill_in_list(skill_name: str, skill_list: list[str]) -> bool:
    norm = _normalise(skill_name)
    return any(kw in norm or norm in kw for kw in skill_list)


# ---------------------------------------------------------------------------
# 1. Skill Score
# ---------------------------------------------------------------------------

def compute_skill_score(
    skills: list[dict[str, Any]],
    assessment_scores: dict[str, float] | None = None,
) -> float:
    """
    Returns a skill score in [0.0, 1.0].

    For each skill:
      - If must-have: weight 1.0 × proficiency_weight × duration_factor
      - If nice-to-have: weight 0.40 × proficiency_weight × duration_factor
      - Assessment score (if present) boosts proficiency_weight slightly

    Duration factor = min(duration_months / 18, 1.0)   (caps at 18 months)
    Final score normalised to [0, 1].
    """
    if not skills:
        return 0.0

    assessment_scores = assessment_scores or {}
    total_weight = 0.0
    weighted_score = 0.0

    # Max possible weighted score for normalisation
    # (assume 6 must-haves at full weight = 6.0, normalise to that)
    NORMALISER = 6.0

    for skill in skills:
        name = skill.get("name", "")
        proficiency = str(skill.get("proficiency", "")).lower()
        duration = skill.get("duration_months", 0) or 0
        endorsements = skill.get("endorsements", 0) or 0

        prof_w = PROFICIENCY_WEIGHTS.get(proficiency, PROFICIENCY_WEIGHTS[""])

        # Assessment score boost: if assessment score exists, blend it in
        assess_key = _normalise(name)
        for k, v in assessment_scores.items():
            if _normalise(k) in assess_key or assess_key in _normalise(k):
                # Blend: 70% proficiency, 30% assessment normalised to [0,1]
                prof_w = 0.70 * prof_w + 0.30 * (v / 100.0)
                break

        # Duration factor (caps at 18 months = 1.0)
        duration_factor = min(duration / 18.0, 1.0) if duration > 0 else 0.15

        # Endorsement micro-boost (caps at 5% bonus)
        endorse_factor = 1.0 + min(endorsements * 0.005, 0.05)

        if _skill_in_list(name, MUST_HAVE_SKILLS):
            item_weight = 1.0
        elif _skill_in_list(name, NICE_TO_HAVE_SKILLS):
            item_weight = 0.40
        else:
            item_weight = 0.05  # irrelevant skills add near-zero

        contribution = item_weight * prof_w * duration_factor * endorse_factor
        weighted_score += contribution
        total_weight += item_weight

    return round(min(weighted_score / NORMALISER, 1.0), 4)


# ---------------------------------------------------------------------------
# 2. Experience & Career Score
# ---------------------------------------------------------------------------

def compute_experience_score(profile: dict[str, Any], career: list[dict[str, Any]]) -> float:
    """
    Returns an experience score in [0.0, 1.0].

    Components:
      - Years of experience fit (JD wants 5–9, peaks at 6–8)
      - Product company bonus (vs pure consulting)
      - Job-hopping penalty
      - Production deployment signal (keyword presence in descriptions)
    """
    yoe = profile.get("years_of_experience", 0) or 0

    # -- Years fit: bell curve peaking at 6-8 years --
    if 6 <= yoe <= 8:
        yoe_score = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        yoe_score = 0.90
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.75
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        yoe_score = 0.55
    elif yoe < 3:
        yoe_score = 0.30
    else:  # >13 years
        yoe_score = 0.60

    # -- Product company bonus: penalise pure consulting career --
    all_consulting = True
    has_any_product = False
    for role in career:
        company_lower = role.get("company", "").lower()
        is_consulting = any(c in company_lower for c in CONSULTING_COMPANIES)
        if not is_consulting:
            all_consulting = False
            has_any_product = True

    consulting_mult = 0.70 if all_consulting else (0.92 if not has_any_product else 1.0)

    # -- Job-hopping penalty: avg tenure < 18 months across 3+ roles --
    tenures = [
        r.get("duration_months", 0)
        for r in career
        if (r.get("duration_months", 0) or 0) > 0
    ]
    job_hop_mult = 1.0
    if len(tenures) >= 3:
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure < 12:
            job_hop_mult = 0.75
        elif avg_tenure < 18:
            job_hop_mult = 0.88

    # -- Production deployment signal: look for keywords in descriptions --
    prod_keywords = [
        "production", "deployed", "real users", "at scale", "millions",
        "latency", "a/b test", "ab test", "online", "inference", "serving",
    ]
    all_descriptions = " ".join(
        (r.get("description", "") or "").lower() for r in career
    )
    prod_hits = sum(1 for kw in prod_keywords if kw in all_descriptions)
    prod_bonus = min(prod_hits * 0.04, 0.20)  # up to +0.20 bonus

    raw_score = (yoe_score + prod_bonus) * consulting_mult * job_hop_mult
    return round(min(raw_score, 1.0), 4)


# ---------------------------------------------------------------------------
# 3. Location Score
# ---------------------------------------------------------------------------

def compute_location_score(
    profile: dict[str, Any],
    signals: dict[str, Any],
) -> float:
    """Returns a location score in [0.0, 1.0]."""
    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)
    work_mode = (signals.get("preferred_work_mode", "") or "").lower()

    # In preferred Indian city
    in_preferred = any(city in location for city in PREFERRED_LOCATIONS)

    # In India but not preferred city
    in_india = "india" in country or any(city in location for city in PREFERRED_LOCATIONS)

    if in_preferred:
        score = 1.0
    elif in_india and willing_relocate:
        score = 0.90
    elif in_india:
        score = 0.75
    elif willing_relocate:
        score = 0.60
    else:
        score = 0.40  # outside India, not willing to relocate

    # Remote/flexible preference: slight boost if willing to be flexible
    if work_mode in ("flexible", "hybrid"):
        score = min(score + 0.05, 1.0)

    return round(score, 4)


# ---------------------------------------------------------------------------
# 4. Education Score
# ---------------------------------------------------------------------------

def compute_education_score(education: list[dict[str, Any]]) -> float:
    """Returns an education score in [0.0, 1.0]."""
    if not education:
        return 0.40  # no education listed → neutral

    best_score = 0.0
    for edu in education:
        tier = str(edu.get("tier", "")).lower()
        field = (edu.get("field_of_study", "") or "").lower()
        degree = (edu.get("degree", "") or "").lower()

        tier_score = EDU_TIER_WEIGHTS.get(tier, 0.50)

        # Field relevance bonus
        relevant_field = any(f in field for f in RELEVANT_FIELDS)
        field_bonus = 0.10 if relevant_field else 0.0

        # Graduate degree bonus
        grad_bonus = 0.05 if any(d in degree for d in ("m.tech", "m.e.", "ms", "mtech", "phd", "m.s")) else 0.0

        score = min(tier_score + field_bonus + grad_bonus, 1.0)
        best_score = max(best_score, score)

    return round(best_score, 4)


# ---------------------------------------------------------------------------
# 5. Role Incoherence Check (keyword stuffer guard)
# ---------------------------------------------------------------------------

UNRELATED_TITLE_KEYWORDS: list[str] = [
    "marketing", "sales", "hr ", "human resource", "recruiter", "accountant",
    "finance", "legal", "content writer", "graphic design", "ui design",
    "ux design", "project manager", "scrum master", "business analyst",
    "operations manager", "supply chain", "procurement",
]

def is_role_incoherent(profile: dict[str, Any], skills: list[dict[str, Any]]) -> bool:
    """
    Returns True if candidate's title is unrelated to AI/ML/Eng but skill list
    is packed with AI keywords — classic keyword stuffer pattern.
    """
    title = (profile.get("current_title", "") or "").lower()
    title_unrelated = any(kw in title for kw in UNRELATED_TITLE_KEYWORDS)

    if not title_unrelated:
        return False

    # Count AI/ML skills claimed
    ai_skill_count = sum(
        1 for s in skills
        if _skill_in_list(s.get("name", ""), MUST_HAVE_SKILLS)
    )
    # If unrelated title but claims 5+ core AI skills → likely stuffer
    return ai_skill_count >= 5


def get_role_coherence_penalty(profile: dict[str, Any], skills: list[dict[str, Any]]) -> float:
    """Returns a penalty multiplier (0.5 if incoherent, 1.0 otherwise)."""
    return 0.50 if is_role_incoherent(profile, skills) else 1.0


# ---------------------------------------------------------------------------
# 6. Composite Score
# ---------------------------------------------------------------------------

def compute_composite_score(
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    location_score: float,
    education_score: float,
    behavioral_multiplier: float,
    coherence_penalty: float = 1.0,
) -> float:
    """
    Final weighted composite score in [0.0, 1.0].

    Weights:
      35% semantic  (embedding similarity to JD)
      25% skill     (must-have + nice-to-have)
      20% experience (years + career quality)
      10% location
      10% education

    Then × behavioral_multiplier × coherence_penalty
    """
    base = (
        0.35 * semantic_score
        + 0.25 * skill_score
        + 0.20 * experience_score
        + 0.10 * location_score
        + 0.10 * education_score
    )
    final = base * behavioral_multiplier * coherence_penalty
    return round(min(max(final, 0.0), 1.0), 6)
