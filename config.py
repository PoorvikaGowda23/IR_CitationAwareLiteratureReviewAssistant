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
