"""
bootstrap_relevant.py
---------------------
For each query in data/queries.csv, retrieve the top-K candidates using
both FAISS (dense) and BM25 (lexical), pool them, and write a review CSV
at data/candidates.csv with columns:
  query_id | rank | paper_id | title | year | score_dense | score_bm25

Open data/candidates.csv in Excel / VS Code, mark which rows are truly
relevant for each query, copy those paper_ids (semicolon-separated) into
the relevant_ids column of data/queries.csv, then run evaluate.py.
"""

import sys
import pandas as pd
from retrieval_baseline import (
    stage1_candidates,
    _init_bm25,
    _PAPERS_DF,
    _BM25_CANDIDATE_PAPERS,
    _BM25_INDEX,
)
from config import DATA_DIR
from schema import PAPER_ID, TITLE, YEAR

QUERIES_CSV    = DATA_DIR / "queries.csv"
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
TOP_K = 30   # candidates per method per query

def main():
    queries_df = pd.read_csv(QUERIES_CSV, dtype=str).fillna("")
    if "query_id" not in queries_df.columns or "query_text" not in queries_df.columns:
        sys.exit("ERROR: queries.csv must have query_id and query_text columns.")

    # BM25 index and candidate list are already built at import time.
    bm25     = _BM25_INDEX
    # _BM25_CANDIDATE_PAPERS is list[dict] with 'paper_id' key
    bm25_ids = [r[PAPER_ID] for r in _BM25_CANDIDATE_PAPERS]

    paper_meta = _PAPERS_DF.set_index(PAPER_ID)[[TITLE, YEAR]]

    rows = []
    for _, q in queries_df.iterrows():
        qid   = q["query_id"].strip()
        qtext = q["query_text"].strip()
        print(f"  Pooling candidates for [{qid}]: {qtext[:60]}")

        # Dense candidates
        dense = {pid: score for pid, score in stage1_candidates(qtext, k=TOP_K)}

        # BM25 candidates
        import numpy as np
        tokens = qtext.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:TOP_K]
        bm25_cands = {bm25_ids[i]: float(bm25_scores[i]) for i in top_bm25_idx}

        # Pool and deduplicate
        all_ids = set(dense) | set(bm25_cands)
        pooled = []
        for pid in all_ids:
            meta  = paper_meta.loc[pid] if pid in paper_meta.index else None
            title = meta[TITLE] if meta is not None else ""
            year  = int(meta[YEAR]) if meta is not None and pd.notna(meta[YEAR]) else ""
            pooled.append({
                "query_id":    qid,
                "paper_id":    pid,
                "title":       title,
                "year":        year,
                "score_dense": round(dense.get(pid, 0.0), 4),
                "score_bm25":  round(bm25_cands.get(pid, 0.0), 4),
            })

        # Sort by dense score descending, then bm25
        pooled.sort(key=lambda x: (x["score_dense"], x["score_bm25"]), reverse=True)
        for rank, r in enumerate(pooled, 1):
            r["rank"] = rank
        rows.extend(pooled)

    out_df = pd.DataFrame(rows, columns=[
        "query_id", "rank", "paper_id", "title", "year", "score_dense", "score_bm25"
    ])
    out_df.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel
    print(f"\nSaved {len(out_df)} candidate rows -> {CANDIDATES_CSV}")
    print("Open that file, mark relevant papers, then paste their paper_ids into queries.csv.")

if __name__ == "__main__":
    main()
