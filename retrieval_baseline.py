"""
Baseline retrieval implementations: dense SPECTER2 FAISS search and lexical BM25Okapi.

Adheres to the shared interface contract:
retrieve(query: str, k: int = 10) -> list[dict]  # [{'paper_id','score','rank','flags'}]

Functions:
- stage1_candidates: retrieves top-k candidates from FAISS, excluding survey papers.
- retrieve_baseline: dense retrieval baseline conforming to the contract.
- retrieve_bm25: lexical BM25Okapi baseline conforming to the contract.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

import config
import embed
import schema

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── 1. Text Tokenizer Helper ───────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace tokenization for BM25.

    Args:
        text: Input text string.

    Returns:
        List of lowercase token strings.
    """
    return text.lower().split()


# ── 2. Corpus & BM25 Cache ─────────────────────────────────────────────────────

# Global module caches
_PAPERS_DF: pd.DataFrame | None = None
_SURVEY_IDS: set[str] = set()
_PAPER_TITLE_MAP: dict[str, str] = {}
_BM25_INDEX: BM25Okapi | None = None
_BM25_CANDIDATE_PAPERS: list[dict[str, Any]] = []


def _load_corpus_metadata() -> None:
    """Load papers DataFrame and build survey exclusion set and title lookup."""
    global _PAPERS_DF, _SURVEY_IDS, _PAPER_TITLE_MAP

    if _PAPERS_DF is not None:
        return

    # Check for papers.parquet first; fallback to seed_papers.parquet
    papers_path = config.DATA_DIR / "papers.parquet"
    if not papers_path.exists():
        papers_path = config.DATA_DIR / "seed_papers.parquet"

    if not papers_path.exists():
        log.warning("No papers parquet file found in %s", config.DATA_DIR)
        return

    _PAPERS_DF = schema.load_papers(papers_path)

    # Identify survey papers to exclude from retrieval results
    _SURVEY_IDS = set(
        _PAPERS_DF[_PAPERS_DF[schema.IS_SURVEY] == True][schema.PAPER_ID]
    )

    # Build paper_id -> title lookup map
    _PAPER_TITLE_MAP = dict(
        zip(_PAPERS_DF[schema.PAPER_ID], _PAPERS_DF[schema.TITLE])
    )


def _init_bm25() -> None:
    """Build BM25Okapi index over non-survey papers and cache it."""
    global _BM25_INDEX, _BM25_CANDIDATE_PAPERS

    if _BM25_INDEX is not None:
        return

    _load_corpus_metadata()
    if _PAPERS_DF is None:
        return

    # Exclude surveys from the BM25 searchable corpus completely
    non_survey_df = _PAPERS_DF[~_PAPERS_DF[schema.PAPER_ID].isin(_SURVEY_IDS)].copy()

    tokenized_corpus: list[list[str]] = []
    _BM25_CANDIDATE_PAPERS = []

    for _, row in non_survey_df.iterrows():
        title = str(row.get(schema.TITLE) or "")
        abstract = str(row.get(schema.ABSTRACT) or "")
        # Combine title and abstract for lexical representation
        full_text = f"{title} {abstract}"
        tokens = tokenize(full_text)
        tokenized_corpus.append(tokens)

        _BM25_CANDIDATE_PAPERS.append({
            schema.PAPER_ID: row[schema.PAPER_ID],
            schema.TITLE: title,
        })

    if tokenized_corpus:
        _BM25_INDEX = BM25Okapi(tokenized_corpus)
        log.info("Built BM25Okapi index with %d non-survey papers", len(tokenized_corpus))


# Initialize metadata and BM25 index at module import time
_load_corpus_metadata()
_init_bm25()


# ── 3. Stage-1 Candidate Retrieval ─────────────────────────────────────────────

def stage1_candidates(
    query: str,
    k: int = config.STAGE1_K,
) -> list[tuple[str, float]]:
    """Retrieve top-k non-survey candidate papers from the FAISS index.

    This function represents the shared stage-1 retrieval pool used by downstream
    re-ranking and graph expansion components.

    Args:
        query: Natural language search query string.
        k:     Number of candidate papers to retrieve (default config.STAGE1_K).

    Returns:
        List of (paper_id, cosine_score) tuples sorted by score descending.
    """
    _load_corpus_metadata()
    index, paper_ids = embed.load_index()

    # Query FAISS with extra headroom to account for excluded survey papers
    search_k = min(len(paper_ids), k + len(_SURVEY_IDS))

    # Embed query using SPECTER2 adhoc query adapter
    query_vec = embed.embed_query(query)

    # Search FAISS inner product index (equivalent to cosine similarity)
    scores, indices = index.search(query_vec, search_k)

    candidates: list[tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(paper_ids):
            continue
        pid = paper_ids[idx]

        # Strictly exclude survey papers
        if pid in _SURVEY_IDS:
            continue

        candidates.append((pid, float(score)))
        if len(candidates) == k:
            break

    return candidates


# ── 4. Dense Retrieval Baseline ────────────────────────────────────────────────

def retrieve_baseline(
    query: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve top-k papers using dense SPECTER2 embeddings + FAISS.

    Adheres strictly to the project retrieval contract:
    [{'paper_id': str, 'score': float, 'rank': int, 'flags': []}]

    Args:
        query: Natural language query string.
        k:     Number of ranked results to return (default 10).

    Returns:
        List of dictionaries conforming to the retrieval interface contract.
    """
    candidates = stage1_candidates(query, k=k)

    results: list[dict[str, Any]] = []
    for rank_idx, (pid, score) in enumerate(candidates[:k], start=1):
        results.append({
            "paper_id": pid,
            "score": float(score),
            "rank": rank_idx,
            "flags": [],  # flags are always empty for baseline
        })

    return results


# ── 5. Lexical BM25 Retrieval Baseline ─────────────────────────────────────────

def retrieve_bm25(
    query: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve top-k papers using BM25Okapi over title + abstract.

    Adheres strictly to the project retrieval contract:
    [{'paper_id': str, 'score': float, 'rank': int, 'flags': []}]

    Args:
        query: Natural language query string.
        k:     Number of ranked results to return (default 10).

    Returns:
        List of dictionaries conforming to the retrieval interface contract.
    """
    _init_bm25()

    if _BM25_INDEX is None or not _BM25_CANDIDATE_PAPERS:
        log.warning("BM25 index is empty. Returning empty list.")
        return []

    tokens = tokenize(query)
    scores = _BM25_INDEX.get_scores(tokens)

    # Rank non-survey papers by BM25 score descending
    num_results = min(k, len(_BM25_CANDIDATE_PAPERS))
    top_indices = np.argsort(scores)[::-1][:num_results]

    results: list[dict[str, Any]] = []
    for rank_idx, idx in enumerate(top_indices, start=1):
        pid = _BM25_CANDIDATE_PAPERS[idx][schema.PAPER_ID]
        score = float(scores[idx])

        results.append({
            "paper_id": pid,
            "score": score,
            "rank": rank_idx,
            "flags": [],
        })

    return results


# ── Main block ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run sample queries and print side-by-side comparison of Dense vs BM25."""
    sample_queries: list[str] = [
        "graph collaborative filtering for recommendation",
        "session-based recommendation using graph neural networks",
        "knowledge graph attention network recommendation",
    ]

    print("\n================ RETRIEVAL BASELINE COMPARISON ================")

    for q_idx, query in enumerate(sample_queries, start=1):
        print(f"\n[Query {q_idx}]: \"{query}\"")
        print("-" * 65)

        # Dense Baseline
        dense_results = retrieve_baseline(query, k=5)
        print("  Dense Baseline (SPECTER2 + FAISS):")
        for res in dense_results:
            title = _PAPER_TITLE_MAP.get(res["paper_id"], "Unknown Title")
            # Sanitize unicode characters for Windows cp1252 terminal compatibility
            safe_title = title.encode("ascii", errors="replace").decode("ascii")
            print(f"    #{res['rank']} [Score: {res['score']:.4f}] {safe_title[:60]}")

        print()

        # BM25 Baseline
        bm25_results = retrieve_bm25(query, k=5)
        print("  Lexical Baseline (BM25Okapi):")
        for res in bm25_results:
            title = _PAPER_TITLE_MAP.get(res["paper_id"], "Unknown Title")
            safe_title = title.encode("ascii", errors="replace").decode("ascii")
            print(f"    #{res['rank']} [Score: {res['score']:.4f}] {safe_title[:60]}")

    print("\n===============================================================\n")


if __name__ == "__main__":
    main()
