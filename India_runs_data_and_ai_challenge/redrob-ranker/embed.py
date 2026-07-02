"""
embed.py — Pre-compute candidate embeddings and save to disk.

This script is ALLOWED to run slowly (outside the 5-min ranking budget).
It embeds all 100K candidates and saves a (100K, embedding_dim) float32 array.

Usage:
    python embed.py --candidates ../candidates.jsonl --out ./embeddings/

Outputs:
    embeddings/candidate_embeddings.npy    — shape (N, 384) float32
    embeddings/candidate_ids.npy           — shape (N,) array of candidate_id strings
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# JD text — what we're matching candidates against
# We use the "must-have" and "nice-to-have" sections from the JD, focused
# on what the JD *means*, not just what it *says*.
# ---------------------------------------------------------------------------
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


def build_candidate_text(candidate: dict) -> str:
    """
    Construct a rich text blob from a candidate record for embedding.
    Prioritises: headline + summary + career descriptions + relevant skills.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    certifications = candidate.get("certifications", [])

    parts = []

    # Headline and summary (highest signal density)
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    # Career descriptions (actual work done, not claimed skills)
    for role in career:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")
        dur = role.get("duration_months", 0) or 0
        if desc:
            # Weight longer tenures more by repeating title context
            prefix = f"{title} at {company} ({dur}mo): " if title else ""
            parts.append(prefix + desc)

    # Skills: name + proficiency + duration
    skill_strs = []
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0) or 0
        if name:
            skill_strs.append(f"{name} ({prof}, {dur}mo)" if prof else name)
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs))

    # Certifications
    cert_names = [c.get("name", "") for c in certifications if c.get("name")]
    if cert_names:
        parts.append("Certifications: " + ", ".join(cert_names))

    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl (or .jsonl.gz)"
    )
    parser.add_argument(
        "--out", default="./embeddings",
        help="Output directory for embeddings (default: ./embeddings)"
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Embedding batch size (default: 256)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "candidate_embeddings.npy"
    ids_path = out_dir / "candidate_ids.npy"
    jd_path = out_dir / "jd_embedding.npy"

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    # Embed JD query first
    print("Embedding JD query...")
    jd_emb = model.encode(JD_QUERY, normalize_embeddings=True, show_progress_bar=False)
    np.save(jd_path, jd_emb)
    print(f"JD embedding saved → {jd_path}")

    # Stream candidates
    print(f"Reading candidates from: {args.candidates}")
    candidates_path = args.candidates

    import gzip
    open_fn = gzip.open if candidates_path.endswith(".gz") else open

    candidate_ids = []
    texts = []

    with open_fn(candidates_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading candidates", unit="cand"):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                cid = cand.get("candidate_id", "")
                text = build_candidate_text(cand)
                candidate_ids.append(cid)
                texts.append(text)
            except json.JSONDecodeError:
                continue

    print(f"Total candidates loaded: {len(texts)}")

    # Batch encode
    print(f"Encoding {len(texts)} candidates in batches of {args.batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Save
    np.save(emb_path, embeddings.astype(np.float32))
    np.save(ids_path, np.array(candidate_ids, dtype=object))

    print(f"\nDone!")
    print(f"  Embeddings: {emb_path}  shape={embeddings.shape}")
    print(f"  IDs:        {ids_path}  count={len(candidate_ids)}")
    print(f"  JD embed:   {jd_path}")


if __name__ == "__main__":
    main()
