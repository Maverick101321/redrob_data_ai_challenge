import json
import gradio as gr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from honeypot import is_honeypot
from behavioral import compute_behavioral_multiplier
from scorer import (
    compute_skill_score,
    compute_experience_score,
    compute_location_score,
    compute_education_score,
    compute_composite_score,
    get_role_coherence_penalty,
)
from reasoning import generate_reasoning

# Load model globally so it caches
print("Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

JD_QUERY = """
Senior AI/ML Engineer role focused on production retrieval and ranking systems.

Requirements:
- Production experience with embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings)
- Vector database operations: Pinecone, Weaviate, Qdrant, FAISS, Milvus, Elasticsearch, OpenSearch
- Strong Python engineering with production deployments to real users at scale
- Evaluation frameworks for ranking: NDCG, MRR, MAP, A/B testing, offline evaluation
- 5-9 years experience at product companies (not pure IT services / consulting)

Nice to have:
- LLM fine-tuning: LoRA, QLoRA, PEFT
- Learning to rank: XGBoost, LightGBM, neural rankers
- RAG systems, distributed inference, MLflow

The ideal candidate has shipped at least one end-to-end ranking or recommendation system
to real users, has strong opinions on dense vs hybrid retrieval, and can evaluate
ranking systems rigorously. Product company experience preferred over IT services.
"""

def build_candidate_text(candidate):
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    certifications = candidate.get('certifications', [])
    
    parts = []
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    
    if headline: parts.append(headline)
    if summary: parts.append(summary)
    
    for role in career:
        title = role.get('title', '')
        company = role.get('company', '')
        desc = role.get('description', '')
        dur = role.get('duration_months', 0) or 0
        if desc:
            prefix = f"{title} at {company} ({dur}mo): " if title else ""
            parts.append(prefix + desc)
            
    skill_strs = []
    for s in skills:
        name = s.get('name', '')
        prof = s.get('proficiency', '')
        dur = s.get('duration_months', 0) or 0
        if name:
            skill_strs.append(f"{name} ({prof}, {dur}mo)" if prof else name)
            
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs))
        
    cert_names = [c.get('name', '') for c in certifications if c.get('name')]
    if cert_names:
        parts.append("Certifications: " + ", ".join(cert_names))
        
    return " | ".join(parts)


def run_ranking():
    # 1. Load candidates
    with open("sample_candidates.json", "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    # 2. Embed JD
    jd_emb = model.encode(JD_QUERY, normalize_embeddings=True)
    
    scored = []
    honeypot_count = 0
    incoherent_count = 0
    
    # 3. Process each candidate
    for cand in candidates:
        cid = cand.get("candidate_id", "")
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        education = cand.get("education", [])
        signals = cand.get("redrob_signals", {})
        assessment = (signals.get("skill_assessment_scores") or {})

        # Honeypot
        hp_flag, hp_reason = is_honeypot(cand)
        if hp_flag:
            honeypot_count += 1
            continue
            
        # Semantic
        text = build_candidate_text(cand)
        cand_emb = model.encode(text, normalize_embeddings=True)
        sem_score = float(np.dot(cand_emb, jd_emb))
        sem_score = (sem_score + 1.0) / 2.0  # scale to 0-1
        
        # Signals
        skill_score = compute_skill_score(skills, assessment)
        exp_score = compute_experience_score(profile, career)
        loc_score = compute_location_score(profile, signals)
        edu_score = compute_education_score(education)
        behav_mult = compute_behavioral_multiplier(signals)
        coherence_penalty = get_role_coherence_penalty(profile, skills)
        
        if coherence_penalty < 1.0:
            incoherent_count += 1
            
        # Composite
        composite = compute_composite_score(
            semantic_score=sem_score,
            skill_score=skill_score,
            experience_score=exp_score,
            location_score=loc_score,
            education_score=edu_score,
            behavioral_multiplier=behav_mult,
            coherence_penalty=coherence_penalty,
        )
        
        scored.append({
            "Candidate ID": cid,
            "Score": round(composite, 4),
            "Semantic": round(sem_score, 4),
            "Skills": round(skill_score, 4),
            "Exp": round(exp_score, 4),
            "Mult": round(behav_mult, 2),
            "Reasoning": generate_reasoning(cand, composite, 1) # rank 1 placeholder for text
        })

    # Sort matches rank.py tie-breaker
    scored.sort(key=lambda x: (-x["Score"], x["Candidate ID"]))
    
    # Add Rank column
    for i, row in enumerate(scored):
        row["Rank"] = i + 1
        
    df = pd.DataFrame(scored)
    # Reorder columns
    df = df[["Rank", "Candidate ID", "Score", "Semantic", "Skills", "Exp", "Mult", "Reasoning"]]
    
    status_msg = f"Processed {len(candidates)} candidates. Filtered {honeypot_count} honeypots. Penalised {incoherent_count} stuffers."
    return df, status_msg


with gr.Blocks(title="Redrob Ranker Sandbox") as app:
    gr.Markdown("# 🚀 Redrob Ranker — Sandbox Demo")
    gr.Markdown("This sandbox runs the exact ranking pipeline on a small sample of 50 candidates from the dataset. It executes embedding generation and our multi-signal composite scoring live.")
    
    run_btn = gr.Button("▶️ Rank Sample Candidates", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    results = gr.Dataframe(label="Top Candidates")
    
    run_btn.click(fn=run_ranking, outputs=[results, status])

if __name__ == "__main__":
    app.launch()
