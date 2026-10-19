"""
sanity_checks.py
----------------
Step 14: Comprehensive sanity checks on the VALIDATION split.

CRITICAL CONSTRAINTS:
  - Uses the VAL split ONLY. Never reads or touches the TEST split.
  - Does NOT alter retrieval.py, tune.py, or config.py.

Checks performed:
  1. Print count of validation queries and relevant IDs per query.
  2. Recall@50 of Stage 1 (stage1_candidates) on validation (per-query and mean ceiling).
  3. Valuation table (nDCG@10 and MAP with 95% bootstrap CIs) for:
       (a) Dense baseline (SPECTER2 + FAISS)
       (b) BM25 baseline
       (c) Stage-1 top-50 reranked by untyped in-degree in corpus graph only
       (d) Stage-1 top-50 reranked by untyped PageRank only
       (e) Random reorder of top 50 (averaged over 20 random seeds)
       (f) Tuned weights from grid search:
           - Row 1: alpha=0.0, beta=0.4, gamma=0.0, delta=0.6
           - Row 3: alpha=0.1, beta=0.0, gamma=0.3, delta=0.6
           - Row 4: alpha=0.1, beta=0.1, gamma=0.3, delta=0.5
       (g) Row 4 evaluated with untyped graph (untyped=True)
  4. Paired bootstrap probability:
       - P(Row 4 > Dense baseline)
       - P(Row 4 > In-degree baseline (c))
  5. Leakage check:
       - Relevant share among candidates with >= 1 graph edge vs without edges, and their ratio.
  6. Graph summary:
       - Total nodes and edges in graph.
       - Share of validation-relevant papers that are graph nodes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from evaluate import (
    GROUND_TRUTH_JSON,
    average_precision,
    bootstrap_ci,
    load_ground_truth,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)
from graph_builder import annotate_graph, build_graph, load_graph
from retrieval import get_stage1_candidates, retrieve
from retrieval_baseline import retrieve_baseline, retrieve_bm25


def run_sanity_checks() -> None:
    # ── 1. Load Validation Split ONLY ─────────────────────────────────────────
    gt_records = load_ground_truth(GROUND_TRUTH_JSON)
    val_records = [r for r in gt_records if r.get("split") == "val"]
    assert len(val_records) > 0, "No validation records found in ground_truth.json!"

    print("\n" + "=" * 70)
    print("SANITY CHECKS (VALIDATION SPLIT ONLY)")
    print("=" * 70)

    print(f"\n[1] Number of Validation Queries: {len(val_records)}")
    for r in val_records:
        qid = r["query_id"]
        qtext = r["query_text"]
        rel_ids = r["relevant_paper_ids"]
        print(f"  • Query [{qid}]: {len(rel_ids)} relevant papers | '{qtext}'")

    # ── 2. Recall@50 of Stage 1 (Ceiling for Reranker) ────────────────────────
    print("\n" + "-" * 70)
    print("[2] Stage 1 Recall@50 on Validation (Ceiling for Reranker)")
    print("-" * 70)
    r50_scores: list[float] = []
    for r in val_records:
        qid = r["query_id"]
        qtext = r["query_text"]
        rel = set(r["relevant_paper_ids"])
        cands = [pid for pid, _ in get_stage1_candidates(qtext, k=50)]
        score = recall_at_k(cands, rel, k=50)
        hits = len(set(cands) & rel)
        r50_scores.append(score)
        print(f"  • [{qid:<12}] Recall@50: {score:>7.4f}  ({hits:>2}/{len(rel):<2} relevant retrieved)")

    mean_r50 = float(np.mean(r50_scores))
    print(f"  --> Mean Stage 1 Recall@50: {mean_r50:.4f} ({mean_r50:.2%})")

    # ── 3. Evaluation on Validation (nDCG@10, MAP with 95% Bootstrap CI) ──────
    print("\n" + "-" * 70)
    print("[3] Evaluating Retrieval Methods on Validation Split")
    print("-" * 70)

    # Load graphs
    G = load_graph(config.DATA_DIR / "graph.gpickle")
    edges_pred_df = pd.read_parquet(config.DATA_DIR / "edges_pred.parquet")
    G_untyped = build_graph(edges_pred_df, untyped=True)
    annotate_graph(G_untyped, seed=config.SEED)

    # Candidate cache for validation queries
    val_stage1_map = {
        r["query_id"]: get_stage1_candidates(r["query_text"], k=50)
        for r in val_records
    }

    # Evaluate helper
    def eval_pipeline(retrieve_func) -> tuple[float, tuple[float, float], float, tuple[float, float], list[float], list[float]]:
        ndcg_list: list[float] = []
        map_list: list[float] = []
        for rec in val_records:
            qtext = rec["query_text"]
            rel = set(rec["relevant_paper_ids"])
            ranked = retrieve_func(qtext, rec)
            ndcg_list.append(ndcg_at_k(ranked, rel, k=10))
            map_list.append(average_precision(ranked, rel))

        mean_ndcg = float(np.mean(ndcg_list))
        ci_ndcg = bootstrap_ci(ndcg_list, n=1000, seed=config.SEED)
        mean_map = float(np.mean(map_list))
        ci_map = bootstrap_ci(map_list, n=1000, seed=config.SEED)
        return mean_ndcg, ci_ndcg, mean_map, ci_map, ndcg_list, map_list

    # (a) Dense baseline
    m_dense, ci_dense_n, map_dense, ci_dense_m, scores_dense_n, scores_dense_m = eval_pipeline(
        lambda q, r: [item["paper_id"] for item in retrieve_baseline(q, k=50)]
    )

    # (b) BM25 baseline
    m_bm25, ci_bm25_n, map_bm25, ci_bm25_m, scores_bm25_n, scores_bm25_m = eval_pipeline(
        lambda q, r: [item["paper_id"] for item in retrieve_bm25(q, k=50)]
    )

    # (c) In-degree rerank of top-50
    def retrieve_indegree(qtext: str, rec: dict) -> list[str]:
        cands = val_stage1_map[rec["query_id"]]
        scored = []
        for pid, emb in cands:
            deg = G.in_degree(pid) if pid in G else 0
            scored.append((deg, emb, pid))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [pid for _, _, pid in scored]

    m_indeg, ci_indeg_n, map_indeg, ci_indeg_m, scores_indeg_n, scores_indeg_m = eval_pipeline(retrieve_indegree)

    # (d) Untyped PageRank rerank of top-50
    def retrieve_untyped_pr(qtext: str, rec: dict) -> list[str]:
        cands = val_stage1_map[rec["query_id"]]
        scored = []
        for pid, emb in cands:
            pr = G_untyped.nodes[pid].get("pagerank", 0.0) if pid in G_untyped else 0.0
            scored.append((pr, emb, pid))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [pid for _, _, pid in scored]

    m_untyped_pr, ci_upr_n, map_untyped_pr, ci_upr_m, scores_upr_n, scores_upr_m = eval_pipeline(retrieve_untyped_pr)

    # (e) Random reorder of top-50 (averaged over 20 seeds)
    rand_ndcg_seeds: list[list[float]] = []
    rand_map_seeds: list[list[float]] = []
    for s in range(config.SEED, config.SEED + 20):
        rng = random.Random(s)
        n_list = []
        m_list = []
        for rec in val_records:
            cands = [p for p, _ in val_stage1_map[rec["query_id"]]]
            shuffled = cands.copy()
            rng.shuffle(shuffled)
            rel = set(rec["relevant_paper_ids"])
            n_list.append(ndcg_at_k(shuffled, rel, k=10))
            m_list.append(average_precision(shuffled, rel))
        rand_ndcg_seeds.append(n_list)
        rand_map_seeds.append(m_list)

    # Average scores per query across seeds
    rand_ndcg_per_q = [float(np.mean([rand_ndcg_seeds[s_idx][q_idx] for s_idx in range(20)])) for q_idx in range(len(val_records))]
    rand_map_per_q = [float(np.mean([rand_map_seeds[s_idx][q_idx] for s_idx in range(20)])) for q_idx in range(len(val_records))]
    m_rand = float(np.mean(rand_ndcg_per_q))
    ci_rand_n = bootstrap_ci(rand_ndcg_per_q, n=1000, seed=config.SEED)
    map_rand = float(np.mean(rand_map_per_q))
    ci_rand_m = bootstrap_ci(rand_map_per_q, n=1000, seed=config.SEED)

    # (f) Tuned weights from grid search: Rows 1, 3, 4
    # Row 1: alpha=0.0, beta=0.4, gamma=0.0, delta=0.6
    w_row1 = {"alpha": 0.0, "beta": 0.4, "gamma": 0.0, "delta": 0.6}
    m_row1, ci_r1_n, map_row1, ci_r1_m, scores_r1_n, scores_r1_m = eval_pipeline(
        lambda q, r: [item["paper_id"] for item in retrieve(q, k=50, weights=w_row1, graph=G, surface_contradictions=False)]
    )

    # Row 3: alpha=0.1, beta=0.0, gamma=0.3, delta=0.6
    w_row3 = {"alpha": 0.1, "beta": 0.0, "gamma": 0.3, "delta": 0.6}
    m_row3, ci_r3_n, map_row3, ci_r3_m, scores_r3_n, scores_r3_m = eval_pipeline(
        lambda q, r: [item["paper_id"] for item in retrieve(q, k=50, weights=w_row3, graph=G, surface_contradictions=False)]
    )

    # Row 4: alpha=0.1, beta=0.1, gamma=0.3, delta=0.5
    w_row4 = {"alpha": 0.1, "beta": 0.1, "gamma": 0.3, "delta": 0.5}
    m_row4, ci_r4_n, map_row4, ci_r4_m, scores_r4_n, scores_r4_m = eval_pipeline(
        lambda q, r: [item["paper_id"] for item in retrieve(q, k=50, weights=w_row4, graph=G, surface_contradictions=False)]
    )

    # (g) Row 4 with graph built with untyped=True
    m_row4_untyped, ci_r4u_n, map_row4_untyped, ci_r4u_m, scores_r4u_n, scores_r4u_m = eval_pipeline(
        lambda q, r: [item["paper_id"] for item in retrieve(q, k=50, weights=w_row4, graph=G_untyped, surface_contradictions=False)]
    )

    # Compile table
    table_rows = [
        ("(a) Dense baseline (SPECTER2 + FAISS)", m_dense, ci_dense_n, map_dense, ci_dense_m),
        ("(b) BM25 baseline", m_bm25, ci_bm25_n, map_bm25, ci_bm25_m),
        ("(c) Stage-1 reranked by untyped in-degree", m_indeg, ci_indeg_n, map_indeg, ci_indeg_m),
        ("(d) Stage-1 reranked by untyped PageRank", m_untyped_pr, ci_upr_n, map_untyped_pr, ci_upr_m),
        ("(e) Random reorder of top-50 (20 seeds)", m_rand, ci_rand_n, map_rand, ci_rand_m),
        ("(f1) Row 1: a=0.0 b=0.4 g=0.0 d=0.6", m_row1, ci_r1_n, map_row1, ci_r1_m),
        ("(f3) Row 3: a=0.1 b=0.0 g=0.3 d=0.6", m_row3, ci_r3_n, map_row3, ci_r3_m),
        ("(f4) Row 4: a=0.1 b=0.1 g=0.3 d=0.5", m_row4, ci_r4_n, map_row4, ci_r4_m),
        ("(g) Row 4 with untyped graph", m_row4_untyped, ci_r4u_n, map_row4_untyped, ci_r4u_m),
    ]

    print(f"{'Method':<44} {'nDCG@10':>8} {'95% CI (nDCG)':>17} {'MAP':>8} {'95% CI (MAP)':>17}")
    print("-" * 98)
    for name, nd, ci_n, mp, ci_m in table_rows:
        ci_n_str = f"[{ci_n[0]:.4f}, {ci_n[1]:.4f}]"
        ci_m_str = f"[{ci_m[0]:.4f}, {ci_m[1]:.4f}]"
        print(f"{name:<44} {nd:>8.4f} {ci_n_str:>17} {mp:>8.4f} {ci_m_str:>17}")

    # ── 4. Paired Bootstrap Significance Tests ────────────────────────────────
    print("\n" + "-" * 70)
    print("[4] Paired Bootstrap Significance Tests")
    print("-" * 70)
    # Probability Row 4 beats Dense baseline
    p_r4_beats_dense_ndcg = paired_bootstrap(scores_r4_n, scores_dense_n, n=1000, seed=config.SEED)
    p_r4_beats_dense_map = paired_bootstrap(scores_r4_m, scores_dense_m, n=1000, seed=config.SEED)
    print(f"  • P(Row 4 > Dense Baseline) [nDCG@10]: {p_r4_beats_dense_ndcg:.4f} ({p_r4_beats_dense_ndcg:.1%})")
    print(f"  • P(Row 4 > Dense Baseline) [MAP]:     {p_r4_beats_dense_map:.4f} ({p_r4_beats_dense_map:.1%})")

    # Probability Row 4 beats (c) in-degree
    p_r4_beats_indeg_ndcg = paired_bootstrap(scores_r4_n, scores_indeg_n, n=1000, seed=config.SEED)
    p_r4_beats_indeg_map = paired_bootstrap(scores_r4_m, scores_indeg_m, n=1000, seed=config.SEED)
    print(f"  • P(Row 4 > In-Degree (c))  [nDCG@10]: {p_r4_beats_indeg_ndcg:.4f} ({p_r4_beats_indeg_ndcg:.1%})")
    print(f"  • P(Row 4 > In-Degree (c))  [MAP]:     {p_r4_beats_indeg_map:.4f} ({p_r4_beats_indeg_map:.1%})")

    # ── 5. Leakage Check ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[5] Leakage Check: Relevance Rates by Graph Edge Presence")
    print("-" * 70)
    with_edge_total = 0
    with_edge_rel = 0
    without_edge_total = 0
    without_edge_rel = 0

    for rec in val_records:
        rel = set(rec["relevant_paper_ids"])
        cands = [pid for pid, _ in val_stage1_map[rec["query_id"]]]
        for pid in cands:
            has_edge = (pid in G and G.degree(pid) > 0)
            is_rel = (pid in rel)
            if has_edge:
                with_edge_total += 1
                if is_rel:
                    with_edge_rel += 1
            else:
                without_edge_total += 1
                if is_rel:
                    without_edge_rel += 1

    rate_with = (with_edge_rel / with_edge_total) if with_edge_total > 0 else 0.0
    rate_without = (without_edge_rel / without_edge_total) if without_edge_total > 0 else 0.0
    ratio = (rate_with / rate_without) if rate_without > 0 else float("inf")

    print(f"  • Candidates WITH >= 1 graph edge : {with_edge_rel:>2}/{with_edge_total:<3} relevant -> {rate_with:.2%}")
    print(f"  • Candidates WITHOUT graph edges  : {without_edge_rel:>2}/{without_edge_total:<3} relevant -> {rate_without:.2%}")
    print(f"  • Ratio (with / without)          : {ratio:.2f}x")

    # ── 6. Graph Nodes & Validation Coverage ──────────────────────────────────
    print("\n" + "-" * 70)
    print("[6] Graph Statistics & Validation Ground-Truth Coverage")
    print("-" * 70)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    all_val_rel = set().union(*(r["relevant_paper_ids"] for r in val_records))
    in_graph = all_val_rel & set(G.nodes)
    coverage = len(in_graph) / len(all_val_rel) if all_val_rel else 0.0

    print(f"  • Total Graph Nodes : {n_nodes}")
    print(f"  • Total Graph Edges : {n_edges}")
    print(f"  • Unique Val-Relevant Papers          : {len(all_val_rel)}")
    print(f"  • Val-Relevant Papers in Graph Nodes  : {len(in_graph)} ({coverage:.2%})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_sanity_checks()
