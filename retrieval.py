"""
retrieval.py
------------
Step 12: Citation-aware hybrid graph re-ranking retrieval pipeline.

Pipeline:
  1. Stage 1: stage1_candidates(query, STAGE1_K) from retrieval_baseline.py.
  2. Anchors: top 5 candidates by embedding score.
  3. Per-candidate signals:
     - embedding score (cosine similarity from Stage 1)
     - PageRank (weighted PageRank from citation graph)
     - proximity to anchors: mean(1 / (1 + d)) for undirected shortest path d <= 3, else 0
     - co-citation with anchors: sum of co-citation counts with the anchors
  4. Min-max normalization for each signal inside the candidate set only.
  5. final = ALPHA*emb + BETA*pagerank + GAMMA*proximity + DELTA*cocitation.
  6. Return project contract format with 'components' breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

import config
from graph_builder import load_graph
from retrieval_baseline import stage1_candidates, retrieve_baseline


# ── Global Module Cache for Graph ─────────────────────────────────────────────
_GRAPH: nx.DiGraph | None = None
_UNDIRECTED_GRAPH: nx.Graph | None = None
_PAPERS_TITLE_MAP: dict[str, str] | None = None


def get_graph(graph_path: Path | str = config.DATA_DIR / "graph.gpickle") -> nx.DiGraph:
    """Lazily load and cache the citation graph."""
    global _GRAPH, _UNDIRECTED_GRAPH
    if _GRAPH is None:
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Citation graph not found at {graph_path}. Run graph_builder.py first.")
        _GRAPH = load_graph(graph_path)
        _UNDIRECTED_GRAPH = _GRAPH.to_undirected()
    return _GRAPH


def get_undirected_graph(graph_path: Path | str = config.DATA_DIR / "graph.gpickle") -> nx.Graph:
    """Return cached undirected projection of the citation graph."""
    global _UNDIRECTED_GRAPH
    if _UNDIRECTED_GRAPH is None:
        get_graph(graph_path)
    assert _UNDIRECTED_GRAPH is not None
    return _UNDIRECTED_GRAPH


def get_title_map(papers_path: Path | str = config.DATA_DIR / "papers.parquet") -> dict[str, str]:
    """Return paper_id -> title mapping."""
    global _PAPERS_TITLE_MAP
    if _PAPERS_TITLE_MAP is None:
        papers_path = Path(papers_path)
        if papers_path.exists():
            pdf = pd.read_parquet(papers_path)
            _PAPERS_TITLE_MAP = dict(zip(pdf["paper_id"].astype(str), pdf["title"].astype(str)))
        else:
            _PAPERS_TITLE_MAP = {}
    return _PAPERS_TITLE_MAP


# ── Signal Computations ───────────────────────────────────────────────────────

def min_max_normalize(values: list[float]) -> list[float]:
    """
    Min-max normalize a list of floats to [0.0, 1.0].
    If all values are identical, returns [0.0] * len(values).
    """
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    diff = max_v - min_v
    if diff > 1e-12:
        return [(v - min_v) / diff for v in values]
    return [0.0] * len(values)


def compute_proximity_to_anchors(
    candidate_id: str,
    anchors: list[str],
    U: nx.Graph,
    anchor_distances: dict[str, dict[str, int]],
    max_hops: int = 3,
) -> float:
    """
    Compute proximity to anchors:
    1 / (1 + shortest undirected path length) averaged over anchors,
    with paths capped at max_hops (default 3), and 0 if unreachable.
    """
    if not anchors:
        return 0.0

    scores: list[float] = []
    for a in anchors:
        if candidate_id == a:
            scores.append(1.0)  # distance 0 -> 1 / (1 + 0) = 1.0
            continue

        # Look up distance from precomputed cutoff distances
        d_map = anchor_distances.get(a, {})
        d = d_map.get(candidate_id)

        if d is not None and d <= max_hops:
            scores.append(1.0 / (1.0 + float(d)))
        else:
            scores.append(0.0)

    return sum(scores) / len(anchors)


def compute_cocitation_with_anchors(
    candidate_id: str,
    anchors: list[str],
    G: nx.DiGraph,
) -> float:
    """
    Compute co-citation count with anchors:
    Sum over anchors of papers citing both candidate and anchor.
    """
    if candidate_id not in G:
        return 0.0

    in_c = set(G.predecessors(candidate_id))
    if not in_c:
        return 0.0

    total_cocit = 0
    for a in anchors:
        if a in G:
            in_a = set(G.predecessors(a))
            total_cocit += len(in_c & in_a)

    return float(total_cocit)


# ── Shared Retrieval Interface ────────────────────────────────────────────────

def retrieve(
    query: str,
    k: int = 10,
    weights: dict[str, float] | None = None,
    graph: nx.DiGraph | None = None,
    stage1_k: int = config.STAGE1_K,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k papers using hybrid graph re-ranking over Stage 1 candidates.

    Pipeline:
      1. Stage 1: call stage1_candidates(query, STAGE1_K).
      2. Anchors: top 5 candidates by embedding score.
      3. For each candidate compute:
         - embedding score (from Stage 1)
         - PageRank (from citation graph)
         - proximity to anchors (capped at 3 hops, 0 if unreachable)
         - co-citation count with anchors
      4. Min-max normalize each signal inside candidate set only.
      5. final = ALPHA*emb + BETA*pagerank + GAMMA*proximity + DELTA*cocitation.
      6. Return contract format: [{'paper_id', 'score', 'rank', 'flags', 'components'}].

    Args:
        query: Natural language query string.
        k: Number of ranked results to return (default 10).
        weights: Optional dictionary overriding {'alpha', 'beta', 'gamma', 'delta'}.
        graph: Optional pre-loaded NetworkX DiGraph (loads graph.gpickle if None).
        stage1_k: Number of stage 1 candidates to fetch (default config.STAGE1_K).

    Returns:
        List of result dictionaries conforming to the retrieval contract.
    """
    # 1. Stage 1 retrieval
    candidates = stage1_candidates(query, k=stage1_k)
    if not candidates:
        return []

    # Load graph & undirected projection
    G = graph if graph is not None else get_graph()
    U = G.to_undirected() if graph is not None else get_undirected_graph()

    # 2. Anchors: top 5 candidates by embedding score
    anchors = [pid for pid, _ in candidates[:5]]

    # Precompute shortest paths up to 3 hops from each anchor for speed
    anchor_distances: dict[str, dict[str, int]] = {}
    for a in anchors:
        if a in U:
            anchor_distances[a] = nx.single_source_shortest_path_length(U, a, cutoff=3)
        else:
            anchor_distances[a] = {}

    # 3. Compute raw signals for each candidate
    raw_emb: list[float] = []
    raw_pr: list[float] = []
    raw_prox: list[float] = []
    raw_cocit: list[float] = []
    cand_ids: list[str] = []

    for pid, emb_score in candidates:
        cand_ids.append(pid)
        raw_emb.append(float(emb_score))

        # PageRank from node attribute in G (0.0 if not in graph)
        pr_score = float(G.nodes[pid].get("pagerank", 0.0)) if pid in G else 0.0
        raw_pr.append(pr_score)

        # Proximity to anchors
        prox = compute_proximity_to_anchors(pid, anchors, U, anchor_distances, max_hops=3)
        raw_prox.append(prox)

        # Co-citation with anchors
        cocit = compute_cocitation_with_anchors(pid, anchors, G)
        raw_cocit.append(cocit)

    # 4. Min-max normalize each signal inside the candidate set only
    norm_emb = min_max_normalize(raw_emb)
    norm_pr = min_max_normalize(raw_pr)
    norm_prox = min_max_normalize(raw_prox)
    norm_cocit = min_max_normalize(raw_cocit)

    # 5. Weights resolution (switchable for ablation studies)
    if weights is None:
        weights = {}
    alpha = float(weights.get("alpha", weights.get("embedding", config.ALPHA)))
    beta = float(weights.get("beta", weights.get("pagerank", config.BETA)))
    gamma = float(weights.get("gamma", weights.get("proximity", config.GAMMA)))
    delta = float(weights.get("delta", weights.get("cocitation", config.DELTA)))

    # Compute final score
    scored_items: list[dict[str, Any]] = []
    for i, pid in enumerate(cand_ids):
        final_score = (
            alpha * norm_emb[i]
            + beta * norm_pr[i]
            + gamma * norm_prox[i]
            + delta * norm_cocit[i]
        )
        scored_items.append({
            "paper_id": pid,
            "final_score": float(final_score),
            "raw_emb": raw_emb[i],
            "components": {
                "embedding": float(round(norm_emb[i], 6)),
                "pagerank": float(round(norm_pr[i], 6)),
                "proximity": float(round(norm_prox[i], 6)),
                "cocitation": float(round(norm_cocit[i], 6)),
            },
        })

    # Sort descending by final_score (break ties with raw embedding score)
    scored_items.sort(key=lambda x: (x["final_score"], x["raw_emb"]), reverse=True)

    # 6. Format to project retrieval contract
    results: list[dict[str, Any]] = []
    for rank_idx, item in enumerate(scored_items[:k], start=1):
        results.append({
            "paper_id": item["paper_id"],
            "score": float(round(item["final_score"], 6)),
            "rank": rank_idx,
            "flags": [],  # no contradiction surfacing yet
            "components": item["components"],
        })

    return results


# ── Comparison Inspection Entrypoint ──────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 12: Hybrid Graph Re-ranking Retrieval")
    parser.add_argument(
        "--query", type=str,
        default="Graph collaborative filtering methods using GNNs",
        help="Query string to evaluate and compare.",
    )
    parser.add_argument("-k", type=int, default=10, help="Number of results to display.")
    args = parser.parse_args()

    title_map = get_title_map()

    print(f"\nEvaluating query: '{args.query}'\n")

    # Run baseline
    baseline_res = retrieve_baseline(args.query, k=args.k)

    # Run graph re-ranking
    reranked_res = retrieve(args.query, k=args.k)

    # Print baseline
    print("=" * 80)
    print(f"BASELINE DENSE RETRIEVAL (Top {args.k})")
    print("=" * 80)
    print(f"{'Rank':<5} {'Score':>8}  {'Paper ID':<42}  {'Title'}")
    print("-" * 80)
    for r in baseline_res:
        t = title_map.get(r["paper_id"], "Unknown")
        print(f"{r['rank']:<5} {r['score']:>8.4f}  {r['paper_id']:<42}  {t[:45]}")

    # Print reranked with components
    print("\n" + "=" * 80)
    print(f"HYBRID GRAPH RE-RANKED RETRIEVAL (Top {args.k})")
    print(f"Weights: ALPHA={config.ALPHA}, BETA={config.BETA}, GAMMA={config.GAMMA}, DELTA={config.DELTA}")
    print("=" * 80)
    print(f"{'Rank':<5} {'Score':>7}  {'Emb':>6} {'PR':>6} {'Prox':>6} {'CoCit':>6}  {'Title'}")
    print("-" * 80)
    for r in reranked_res:
        c = r["components"]
        t = title_map.get(r["paper_id"], "Unknown")
        print(
            f"{r['rank']:<5} {r['score']:>7.4f}  "
            f"{c['embedding']:>6.3f} {c['pagerank']:>6.3f} {c['proximity']:>6.3f} {c['cocitation']:>6.3f}  "
            f"{t[:45]}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
