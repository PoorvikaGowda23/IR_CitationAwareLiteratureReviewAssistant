"""
Global constants and configuration for the citation-aware literature review assistant.
All other modules import from here; never hard-code these values elsewhere.
"""

import os
from pathlib import Path


def _load_dotenv(filepath: Path = Path(".env")) -> None:
    """Read key-value pairs from a .env file and populate os.environ if not already set.

    Args:
        filepath: Path to the .env file.
    """
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()  # remove leading/trailing whitespace
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)  # split only on first '='
                    os.environ.setdefault(key.strip(), val.strip())


# Load environment variables from .env if present
_load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
# Root of all raw/processed data files (Parquet, FAISS index, etc.)
DATA_DIR: Path = Path("data")

# Root of all evaluation output files (metric JSON, plots, etc.)
RESULTS_DIR: Path = Path("results")

# ── Reproducibility ────────────────────────────────────────────────────────────
# Fixed random seed used by numpy, torch, and sklearn wherever randomness appears
SEED: int = 42

# ── Semantic Scholar API ───────────────────────────────────────────────────────
# Read from the environment; keep the actual key out of source control
S2_API_KEY: str | None = os.environ.get("S2_API_KEY")

# ── Retrieval hyper-parameters ─────────────────────────────────────────────────
# Number of candidates fetched in the first (cheap) retrieval stage
# before re-ranking or graph expansion
STAGE1_K: int = 50

# ── Model identifiers (HuggingFace hub) ───────────────────────────────────────
# SPECTER2 base encoder used to embed paper titles + abstracts
SPECTER_BASE: str = "allenai/specter2_base"

# SciBERT used for intent classification of citation sentences
SCIBERT: str = "allenai/scibert_scivocab_uncased"

# ── Corpus filtering & landmark papers ─────────────────────────────────────────
CORPUS_CAP: int = 1500   # maximum total papers in the 1-hop expanded corpus
MIN_CITATIONS: int = 20  # minimum citationCount required for generic search results

# Known survey papers in the target research domain
SURVEY_TITLES: list[str] = [
    "Graph Neural Networks in Recommender Systems: A Survey",
    "A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions",
    "Graph Learning based Recommender Systems: A Review",
]

# Core landmark / foundational papers for the domain (always included)
SEED_TITLES: list[str] = [
    "Neural Graph Collaborative Filtering",
    "Session-based Recommendation with Graph Neural Networks",
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation",
    "Graph Convolutional Neural Networks for Web-Scale Recommender Systems",
    "KGAT: Knowledge Graph Attention Network for Recommendation",
    "Self-supervised Graph Learning for Recommendation",
    "Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis",
]

# ── Citation-intent weights ────────────────────────────────────────────────────
# Scalar multiplier applied to each citation-intent type during graph re-ranking.
# "background" citations are down-weighted because they are less specific;
# all other intents (method, result, extension, contradiction) are kept at 1.0.
INTENT_WEIGHTS: dict[str, float] = {
    "background":    0.3,   # general context citations, low specificity
    "method":        1.0,   # paper uses the cited work's method
    "result":        1.0,   # paper compares/extends results
    "extension":     1.0,   # paper directly extends the cited work
    "contradiction": 1.0,   # paper disputes or contrasts findings
}

# ── Ground-truth builder behaviour ────────────────────────────────────────────
# Minimum corpus-present relevant IDs required to keep a query in ground truth.
MIN_RELEVANT: int = 8

# When a query row in queries.csv has an empty relevant_ids column, the fallback
# is to use the full survey reference list from survey_refs.json.
# Set to True  → allow the fallback (print a WARNING, then proceed).
# Set to False → reject the query and drop it instead of using the fallback.
ALLOW_SURVEY_FALLBACK: bool = False

# Semantic Scholar stores the first-version year of a survey, but its reference list comes from the latest revision.
YEAR_TOLERANCE: int = 3

# ── Retrieval Re-ranking Weights (Step 12) ───────────────────────────────────
# Linear combination weights for hybrid graph re-ranking:
# final = ALPHA*emb + BETA*pagerank + GAMMA*proximity + DELTA*cocitation
ALPHA: float = 0.1  # embedding score weight (Row 4 of results/tuning_val.csv; keeps the hybrid)
BETA: float = 0.1   # graph PageRank weight
GAMMA: float = 0.3  # anchor proximity weight
DELTA: float = 0.5  # anchor co-citation weight

