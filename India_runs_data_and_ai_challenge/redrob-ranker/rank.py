"""
rank.py — Main ranking entrypoint. Must complete in ≤5 minutes on 16GB CPU.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Requirements:
    - Pre-computed embeddings in ./embeddings/ (run embed.py first)
    - No GPU, no network calls, no hosted LLM APIs during this step

Pipeline:
    1. Load pre-computed embeddings → cosine similarity scores (semantic_score)
    2. Stream candidates.jsonl → compute skill/exp/location/edu/behavioral scores
    3. Detect + hard-filter honeypots
    4. Compute composite scores
    5. Sort → top 100
    6. Generate reasoning
    7. Write submission.csv
"""

import argparse
import json
import gzip
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = Path("./embeddings")
TOP_N = 100


def load_embeddings(emb_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed embeddings and JD embedding from disk."""
    emb_path = emb_dir / "candidate_embeddings.npy"
    ids_path = emb_dir / "candidate_ids.npy"
    jd_path = emb_dir / "jd_embedding.npy"

    for p in [emb_path, ids_path, jd_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing embedding file: {p}\n"
                "Run: python embed.py --candidates ../candidates.jsonl --out ./embeddings/"
            )

    print("Loading embeddings from disk...")
    embeddings = np.load(emb_path)          # shape: (N, 384)
    candidate_ids = np.load(ids_path, allow_pickle=True)  # shape: (N,)
    jd_embedding = np.load(jd_path)          # shape: (384,)

    print(f"  Embeddings: {embeddings.shape}, JD: {jd_embedding.shape}")
    return embeddings, candidate_ids, jd_embedding


def compute_semantic_scores(
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """
    Vectorised cosine similarity between JD and all candidates.
    Both are already L2-normalised (normalise_embeddings=True in embed.py),
    so dot product == cosine similarity.
    Returns shape (N,) float32.
    """
    print("Computing semantic similarity scores...")
    scores = embeddings @ jd_embedding  # dot product, vectorised, very fast
    # Shift from [-1,1] to [0,1]
    scores = (scores + 1.0) / 2.0
    return scores.astype(np.float32)


def rank_candidates(
    candidates_path: str,
    semantic_scores: np.ndarray,
    id_to_idx: dict[str, int],
) -> list[dict]:
    """
    Stream candidates.jsonl, compute all signals, build scored records.
    Returns list of dicts sorted by composite_score descending.
    """
    print("Scoring candidates...")
    open_fn = gzip.open if candidates_path.endswith(".gz") else open

    scored = []
    honeypot_count = 0
    incoherent_count = 0

    with open_fn(candidates_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scoring", unit="cand"):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = cand.get("candidate_id", "")
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])
            education = cand.get("education", [])
            signals = cand.get("redrob_signals", {})
            assessment = (signals.get("skill_assessment_scores") or {})

            # --- 1. Honeypot check (hard filter) ---
            hp_flag, hp_reason = is_honeypot(cand)
            if hp_flag:
                honeypot_count += 1
                continue  # never in top 100

            # --- 2. Semantic score from pre-computed embeddings ---
            idx = id_to_idx.get(cid)
            if idx is not None:
                sem_score = float(semantic_scores[idx])
            else:
                sem_score = 0.30  # missing embedding → low semantic score

            # --- 3. Signal scores ---
            skill_score = compute_skill_score(skills, assessment)
            exp_score = compute_experience_score(profile, career)
            loc_score = compute_location_score(profile, signals)
            edu_score = compute_education_score(education)
            behav_mult = compute_behavioral_multiplier(signals)
            coherence_penalty = get_role_coherence_penalty(profile, skills)

            if coherence_penalty < 1.0:
                incoherent_count += 1

            # --- 4. Composite ---
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
                "candidate_id": cid,
                "composite_score": composite,
                "semantic_score": sem_score,
                "skill_score": skill_score,
                "experience_score": exp_score,
                "location_score": loc_score,
                "education_score": edu_score,
                "behavioral_multiplier": behav_mult,
                "coherence_penalty": coherence_penalty,
                "_candidate": cand,  # keep for reasoning generation
            })

    print(f"  Honeypots filtered: {honeypot_count}")
    print(f"  Incoherent titles penalised: {incoherent_count}")
    print(f"  Remaining candidates: {len(scored)}")

    # Sort by composite descending
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


def write_submission(top100: list[dict], out_path: str):
    """Write submission.csv in the required format."""
    print(f"Writing submission to: {out_path}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, record in enumerate(top100, start=1):
            cand = record["_candidate"]
            cid = record["candidate_id"]
            score = record["composite_score"]

            reasoning = generate_reasoning(
                candidate=cand,
                composite_score=score,
                rank=rank,
            )

            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"  Wrote {len(top100)} rows.")


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob hackathon challenge"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl (or .jsonl.gz)"
    )
    parser.add_argument(
        "--out", default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)"
    )
    parser.add_argument(
        "--embeddings-dir", default="./embeddings",
        help="Directory with pre-computed embeddings (default: ./embeddings)"
    )
    args = parser.parse_args()

    t_start = datetime.now()
    print(f"=== Redrob Ranker — started at {t_start.strftime('%H:%M:%S')} ===\n")

    emb_dir = Path(args.embeddings_dir)

    # Load embeddings
    embeddings, candidate_ids, jd_embedding = load_embeddings(emb_dir)

    # Build ID → index lookup
    id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}

    # Compute semantic scores (vectorised, <1s for 100K)
    semantic_scores = compute_semantic_scores(embeddings, jd_embedding)

    # Score all candidates
    scored = rank_candidates(args.candidates, semantic_scores, id_to_idx)

    # Top 100
    top100 = scored[:TOP_N]
    print(f"\nTop candidate score: {top100[0]['composite_score']:.4f}")
    print(f"Rank 100 score:      {top100[-1]['composite_score']:.4f}")

    # Write submission
    write_submission(top100, args.out)

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Submission: {args.out}")
    print(f"\nNext step: python validate_submission.py {args.out} {args.candidates}")


if __name__ == "__main__":
    main()
