"""
graph_builder.py
----------------
Step 11: Citation graph construction, metrics, community detection, and serialization.

Functions:
  1. build_graph(edges_pred_df, untyped=False): NetworkX DiGraph (citing_id -> cited_id),
     intent weights from config.INTENT_WEIGHTS, collapsing duplicate pairs to max weight.
     Asserts no survey paper is a node.
  2. compute_pagerank(G): Weighted PageRank with alpha=0.85.
  3. cocitation_matrix(G): Sparse matrix where entry (i, j) is the count of papers citing both i and j.
  4. coupling_matrix(G): Sparse matrix where entry (i, j) is the count of references i and j share.
  5. louvain_communities(G): Louvain community detection on undirected weighted projection, seeded.
  6. save_graph(G, path) & load_graph(path): Serialize to data/graph.gpickle with centrality & community attributes.
  7. print_summary_stats(): Nodes, edges, edges per intent, top 10 papers by PageRank with titles,
     number of communities and sizes of the largest 5.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities as nx_louvain
import pandas as pd
import scipy.sparse as sp

from config import DATA_DIR, INTENT_WEIGHTS, SEED

GRAPH_PATH = DATA_DIR / "graph.gpickle"
PAPERS_PATH = DATA_DIR / "papers.parquet"
EDGES_PRED_PATH = DATA_DIR / "edges_pred.parquet"


# ── Survey Paper Identification ───────────────────────────────────────────────

def get_survey_ids(papers_path: Path | str = PAPERS_PATH) -> set[str]:
    """Return the set of paper IDs marked as surveys in papers.parquet."""
    papers_path = Path(papers_path)
    if not papers_path.exists():
        return set()
    try:
        pdf = pd.read_parquet(papers_path)
        if "is_survey" in pdf.columns:
            return set(pdf[pdf["is_survey"]]["paper_id"].astype(str))
    except Exception:
        pass
    return set()


# ── 1. Graph Construction ─────────────────────────────────────────────────────

def build_graph(
    edges_pred_df: pd.DataFrame,
    untyped: bool = False,
    survey_ids: set[str] | None = None,
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from citing_id to cited_id.

    Edge attributes:
      - 'intent': use 'ext_intent' if extension or contradiction with confidence > 0.6,
                  otherwise 'predicted_intent'.
      - 'weight': INTENT_WEIGHTS[intent], or 1.0 if untyped=True.

    Multiple edges between the same pair collapse into one, keeping the highest weight.
    Asserts that no survey paper is a node.

    Args:
        edges_pred_df: DataFrame containing at least ['citing_id', 'cited_id'].
        untyped: If True, set every edge weight to 1.0 (for ablation study).
        survey_ids: Set of survey paper IDs to exclude. If None, loaded from papers.parquet.

    Returns:
        nx.DiGraph: Directed citation graph.
    """
    if survey_ids is None:
        survey_ids = get_survey_ids()

    G = nx.DiGraph()

    for _, row in edges_pred_df.iterrows():
        citing_id = str(row["citing_id"]).strip()
        cited_id = str(row["cited_id"]).strip()

        # Exclude survey papers from being nodes
        if citing_id in survey_ids or cited_id in survey_ids:
            continue

        # Resolve intent
        ext_intent = (
            str(row["ext_intent"]).strip().lower()
            if "ext_intent" in row and pd.notna(row["ext_intent"])
            else None
        )
        ext_conf = float(row.get("ext_confidence", row.get("intent_confidence", 0.0)))

        if ext_intent in ("extension", "contradiction") and ext_conf > 0.6:
            intent = ext_intent
        else:
            intent = str(row.get("predicted_intent", "background")).strip().lower()

        # Determine weight
        if untyped:
            weight = 1.0
        else:
            weight = float(INTENT_WEIGHTS.get(intent, 1.0))

        # Collapse duplicate directed edges: keep the one with higher weight
        if G.has_edge(citing_id, cited_id):
            if weight > G[citing_id][cited_id]["weight"]:
                G[citing_id][cited_id]["weight"] = weight
                G[citing_id][cited_id]["intent"] = intent
        else:
            G.add_edge(citing_id, cited_id, weight=weight, intent=intent)

    # 7. Assert that no survey paper is a node
    survey_nodes = set(G.nodes) & survey_ids
    assert not survey_nodes, (
        f"Assertion failed: survey paper(s) found in graph nodes: {survey_nodes}"
    )

    return G


# ── 2. Weighted PageRank ──────────────────────────────────────────────────────

def compute_pagerank(G: nx.DiGraph, alpha: float = 0.85) -> dict[str, float]:
    """
    Compute weighted PageRank on G with alpha=0.85.

    Args:
        G: Directed citation graph with 'weight' edge attributes.
        alpha: PageRank damping factor (default: 0.85).

    Returns:
        dict[str, float]: Map from node_id to PageRank score.
    """
    if len(G) == 0:
        return {}
    return nx.pagerank(G, alpha=alpha, weight="weight")


# ── 3. Co-citation Matrix ─────────────────────────────────────────────────────

def cocitation_matrix(
    G: nx.DiGraph,
    nodelist: list[str] | None = None,
) -> sp.csr_matrix:
    """
    Compute co-citation sparse matrix C where entry (i, j) is the number of papers citing both i and j.

    Let A be the binary adjacency matrix where A[k, i] = 1 if paper k cites paper i.
    Then C = A^T @ A, where C[i, j] = sum_k (A[k, i] * A[k, j]).

    Args:
        G: Directed citation graph.
        nodelist: Node ordering. Defaults to list(G.nodes()).

    Returns:
        scipy.sparse.csr_matrix: Co-citation matrix.
    """
    if len(G) == 0:
        return sp.csr_matrix((0, 0))
    if nodelist is None:
        nodelist = list(G.nodes())

    A = nx.adjacency_matrix(G, nodelist=nodelist, weight=None)
    C = (A.T @ A).tocsr()
    return C


# ── 4. Bibliographic Coupling Matrix ──────────────────────────────────────────

def coupling_matrix(
    G: nx.DiGraph,
    nodelist: list[str] | None = None,
) -> sp.csr_matrix:
    """
    Compute bibliographic coupling sparse matrix B where entry (i, j) is the number of references i and j share.

    Let A be the binary adjacency matrix where A[i, k] = 1 if paper i cites paper k.
    Then B = A @ A^T, where B[i, j] = sum_k (A[i, k] * A[j, k]).

    Args:
        G: Directed citation graph.
        nodelist: Node ordering. Defaults to list(G.nodes()).

    Returns:
        scipy.sparse.csr_matrix: Bibliographic coupling matrix.
    """
    if len(G) == 0:
        return sp.csr_matrix((0, 0))
    if nodelist is None:
        nodelist = list(G.nodes())

    A = nx.adjacency_matrix(G, nodelist=nodelist, weight=None)
    B = (A @ A.T).tocsr()
    return B


# ── 5. Louvain Community Detection ────────────────────────────────────────────

def louvain_communities(
    G: nx.DiGraph | nx.Graph,
    seed: int = SEED,
) -> list[set[str]]:
    """
    Run Louvain community detection on the undirected weighted projection of G, seeded.

    Args:
        G: Directed or undirected citation graph.
        seed: Random seed for reproducibility.

    Returns:
        list[set[str]]: Communities as lists of sets of node IDs.
    """
    if len(G) == 0:
        return []

    # Undirected weighted projection
    if G.is_directed():
        U = nx.Graph()
        for n, d in G.nodes(data=True):
            U.add_node(n, **d)
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            if U.has_edge(u, v):
                U[u][v]["weight"] = max(U[u][v]["weight"], w)
            else:
                U.add_edge(u, v, weight=w)
    else:
        U = G

    comms = nx_louvain(U, weight="weight", seed=seed)
    return comms


# ── 6. Save & Load Graph ──────────────────────────────────────────────────────

def annotate_graph(
    G: nx.DiGraph,
    pr: dict[str, float] | None = None,
    comms: list[set[str]] | None = None,
    seed: int = SEED,
) -> nx.DiGraph:
    """Attach PageRank centrality and community ID as node attributes."""
    if pr is None:
        pr = compute_pagerank(G)
    if comms is None:
        comms = louvain_communities(G, seed=seed)

    # Attach centrality scores
    for node, score in pr.items():
        G.nodes[node]["pagerank"] = score
        G.nodes[node]["centrality"] = score

    # Attach community IDs
    for c_id, comm in enumerate(comms):
        for node in comm:
            G.nodes[node]["community"] = c_id
            G.nodes[node]["community_id"] = c_id

    return G


def save_graph(
    G: nx.DiGraph,
    path: Path | str = GRAPH_PATH,
    annotate: bool = True,
    seed: int = SEED,
) -> None:
    """
    Save graph to path (default data/graph.gpickle) with centrality scores and community IDs as node attributes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if annotate and len(G) > 0:
        first_node = next(iter(G.nodes))
        if "pagerank" not in G.nodes[first_node] or "community" not in G.nodes[first_node]:
            annotate_graph(G, seed=seed)

    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path: Path | str = GRAPH_PATH) -> nx.DiGraph:
    """Load graph from path (default data/graph.gpickle)."""
    path = Path(path)
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


# ── Summary Stats ─────────────────────────────────────────────────────────────

def print_summary_stats(
    G: nx.DiGraph,
    pr: dict[str, float] | None = None,
    comms: list[set[str]] | None = None,
    papers_path: Path | str = PAPERS_PATH,
) -> None:
    """Print comprehensive summary statistics for the citation graph."""
    if pr is None:
        pr = compute_pagerank(G)
    if comms is None:
        comms = louvain_communities(G)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print("\n" + "=" * 65)
    print("CITATION GRAPH SUMMARY STATISTICS")
    print("=" * 65)
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")

    # Edges per intent
    intent_counts = Counter(d.get("intent", "unknown") for _, _, d in G.edges(data=True))
    print("\nEdges per intent:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = (count / n_edges * 100) if n_edges > 0 else 0
        print(f"  {intent:<15} {count:>6}  ({pct:>5.1f}%)")

    # Top 10 papers by PageRank with titles
    title_map = {}
    papers_path = Path(papers_path)
    if papers_path.exists():
        try:
            pdf = pd.read_parquet(papers_path)
            title_map = dict(zip(pdf["paper_id"].astype(str), pdf["title"].astype(str)))
        except Exception:
            pass

    top10 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 papers by PageRank:")
    print(f"  {'Rank':<5} {'PageRank':>10}  {'Paper Title'}")
    print("  " + "-" * 58)
    for rank, (pid, score) in enumerate(top10, 1):
        title = title_map.get(pid, pid)
        print(f"  {rank:<5} {score:>10.4f}  {title[:58]}")

    # Communities
    n_comms = len(comms)
    comm_sizes = sorted([len(c) for c in comms], reverse=True)
    largest_5 = comm_sizes[:5]
    print(f"\nNumber of communities: {n_comms}")
    print(f"Sizes of largest 5:    {largest_5}")
    print("=" * 65 + "\n")


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11: Citation Graph Builder")
    parser.add_argument(
        "--edges-path", type=Path, default=EDGES_PRED_PATH,
        help=f"Input edges parquet path (default: {EDGES_PRED_PATH})",
    )
    parser.add_argument(
        "--output-path", type=Path, default=GRAPH_PATH,
        help=f"Output gpickle graph path (default: {GRAPH_PATH})",
    )
    parser.add_argument(
        "--untyped", action="store_true",
        help="Build untyped graph where all weights are 1.0 (ablation)",
    )
    args = parser.parse_args()

    print(f"Loading edges from {args.edges_path} ...")
    edges_df = pd.read_parquet(args.edges_path)

    print(f"Building {'untyped' if args.untyped else 'intent-weighted'} graph ...")
    G = build_graph(edges_df, untyped=args.untyped)

    print("Computing weighted PageRank (alpha=0.85) ...")
    pr = compute_pagerank(G, alpha=0.85)

    print("Detecting Louvain communities (seeded) ...")
    comms = louvain_communities(G, seed=SEED)

    print("Annotating node attributes (pagerank, community) ...")
    annotate_graph(G, pr=pr, comms=comms)

    print(f"Saving graph -> {args.output_path} ...")
    save_graph(G, path=args.output_path, annotate=False)

    # Validate roundtrip load
    G_loaded = load_graph(args.output_path)
    assert len(G_loaded) == len(G), "Loaded graph node count mismatch!"

    # Compute co-citation and coupling matrices smoke check
    C = cocitation_matrix(G)
    B = coupling_matrix(G)
    print(f"Co-citation matrix shape: {C.shape}, non-zeros: {C.nnz}")
    print(f"Coupling matrix shape:    {B.shape}, non-zeros: {B.nnz}")

    # Print summary statistics
    print_summary_stats(G, pr=pr, comms=comms)


if __name__ == "__main__":
    main()
