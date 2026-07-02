# Redrob Intelligent Candidate Ranking

AI-powered candidate ranker for the [Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge](https://redrob.ai).

## How it works

A two-phase pipeline:

1. **Pre-computation** (`embed.py`) — embeds all 100K candidates using `all-MiniLM-L6-v2` (offline, once)
2. **Ranking** (`rank.py`) — loads embeddings, computes multi-signal composite scores, outputs top-100 CSV in ≤5 min on CPU

### Scoring formula
```
composite = (
  0.35 × semantic_score      # JD ↔ profile cosine similarity
+ 0.25 × skill_score         # JD must-haves + nice-to-haves with duration weighting
+ 0.20 × experience_score    # years, product-company quality, production signals
+ 0.10 × location_score      # preferred Indian cities, relocation willingness
+ 0.10 × education_score     # institution tier + field relevance
) × behavioral_multiplier     # 0.3–1.0 from 23 Redrob signals
  × coherence_penalty         # keyword-stuffer guard
```

Honeypot profiles (impossible timelines, mass expert-zero-duration skills) are hard-filtered before ranking.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Pre-compute embeddings (once, can be slow)
```bash
python embed.py --candidates ./candidates.jsonl --out ./embeddings/
```

### Step 2: Generate ranked submission (≤5 min, CPU-only)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Step 3: Validate format
```bash
python validate_submission.py submission.csv candidates.jsonl
```

## Compute constraints met
- ✅ No GPU required
- ✅ No network calls during ranking step
- ✅ No hosted LLM APIs
- ✅ Runs in <5 min on 16GB CPU (ranking step only; pre-computation is separate)
- ✅ 16GB RAM sufficient

## Project structure
```
redrob-ranker/
├── rank.py              # Main entrypoint — produces submission.csv
├── embed.py             # Pre-computation — embeds all candidates
├── honeypot.py          # Detects & filters impossible/fraudulent profiles
├── behavioral.py        # 23 Redrob signals → behavioral multiplier
├── scorer.py            # Skill, experience, location, education scoring
├── reasoning.py         # Generates specific per-candidate reasoning text
├── submission_metadata.yaml
├── requirements.txt
└── README.md
```
