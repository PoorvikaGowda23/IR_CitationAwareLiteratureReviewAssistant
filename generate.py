"""
generate.py
-----------
Step 14b: Citation-aware review generator.

Implements generate_review(query, ranked) with the same shared contract as
generate_baseline.py, but builds a richer, intent-structured prompt.

Pipeline:
  1. Load the citation graph and look up the Louvain community_id for each
     retrieved paper. Papers not in the graph go to cluster 'other'.
  2. For each cluster build structured records:
       paper_id, year, tldr, one citation-context sentence, and the intent of
       the strongest graph edge connecting it to another paper in the retrieved set.
  3. Build an intent-ordered prompt:
       - background  -> opens each cluster section
       - method / extension -> continues it
       - contradiction -> contrasts findings (explicitly flagged)
       - result -> compares results
     Instruct the LLM to write ONE CLAIM PER SENTENCE and end every sentence
     with [S2:paper_id].
  4. Post-process identically to generate_baseline: sentence split, regex tag
     extraction, invalid tag check, return {'text', 'claims', ...}.

Same LLM (openai/gpt-oss-120b), temperature=0, max_tokens=2048, disk cache.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Shared utilities re-used from generate_baseline
from generate_baseline import (
    CACHE_DIR,
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY_ENV,
    _build_prompt as _baseline_build_prompt,  # noqa: F401 (not used directly)
    _cache_key,
    _call_groq,
    _extract_claims,
    _load_cache,
    _load_papers_lookup,
    _save_cache,
)

import config
import networkx as nx
from graph_builder import load_graph
import os

# ── Intent ordering for structured prompt sections ────────────────────────────
# Lower number = appears earlier in the review section
_INTENT_ORDER: dict[str, int] = {
    "background":    0,
    "method":        1,
    "extension":     2,
    "result":        3,
    "contradiction": 4,
}
_INTENT_ROLE: dict[str, str] = {
    "background":    "provides background context",
    "method":        "introduces or extends a method",
    "extension":     "directly extends another retrieved paper",
    "result":        "compares or contrasts empirical results",
    "contradiction": "contradicts or challenges findings of another paper",
}

# ── Graph loading (lazy, module-level cache) ───────────────────────────────────
_GRAPH_CACHE: dict[str, nx.DiGraph] = {}


def _get_graph(graph_path: Path = config.DATA_DIR / "graph.gpickle") -> nx.DiGraph:
    key = str(graph_path)
    if key not in _GRAPH_CACHE:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Citation graph not found at {graph_path}. Run graph_builder.py first."
            )
        _GRAPH_CACHE[key] = load_graph(graph_path)
    return _GRAPH_CACHE[key]


# ── Step 1: cluster retrieved papers by community ──────────────────────────────

def _cluster_by_community(
    ranked_ids: list[str],
    G: nx.DiGraph,
) -> dict[str | int, list[str]]:
    """Group retrieved paper IDs by their Louvain community_id.

    Papers not present as nodes in G are placed in the 'other' cluster.

    Args:
        ranked_ids: Ordered list of paper IDs from the retrieval result.
        G: Citation graph with 'community_id' node attributes.

    Returns:
        dict mapping community_id (int or 'other') -> list[paper_id].
        Preserves ranked order within each cluster.
    """
    clusters: dict[str | int, list[str]] = defaultdict(list)
    for pid in ranked_ids:
        if pid in G and "community_id" in G.nodes[pid]:
            cid = int(G.nodes[pid]["community_id"])
        else:
            cid = "other"
        clusters[cid].append(pid)
    return dict(clusters)


# ── Step 2: build structured records for each paper ───────────────────────────

def _dominant_intra_edge(
    pid: str,
    retrieved_set: set[str],
    G: nx.DiGraph,
) -> tuple[str, str]:
    """Return (intent, neighbour_id) for the strongest edge linking pid to
    another paper in the retrieved set.  Falls back to ('background', '') if no
    such edge exists or pid is not in the graph.

    Checks both outgoing (pid -> v) and incoming (u -> pid) edges.
    'Strongest' = highest edge weight; ties broken by intent priority.
    """
    if pid not in G:
        return "background", ""

    best_weight = -1.0
    best_intent = "background"
    best_neighbour = ""

    for v in G.successors(pid):
        if v in retrieved_set and v != pid:
            edata = G[pid][v]
            w = float(edata.get("weight", 1.0))
            intent = str(edata.get("intent", "background")).lower()
            if w > best_weight or (w == best_weight and _INTENT_ORDER.get(intent, 99)
                                   < _INTENT_ORDER.get(best_intent, 99)):
                best_weight = w
                best_intent = intent
                best_neighbour = v

    for u in G.predecessors(pid):
        if u in retrieved_set and u != pid:
            edata = G[u][pid]
            w = float(edata.get("weight", 1.0))
            intent = str(edata.get("intent", "background")).lower()
            if w > best_weight or (w == best_weight and _INTENT_ORDER.get(intent, 99)
                                   < _INTENT_ORDER.get(best_intent, 99)):
                best_weight = w
                best_intent = intent
                best_neighbour = u

    return best_intent, best_neighbour


_EDGE_CONTEXT_CACHE: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}


def _edge_intent(row: Any) -> str:
    """Intent of one edges_pred row, using the same rule as graph_builder.build_graph."""
    ext = row.get("ext_intent")
    ext_conf = row.get("ext_confidence")
    if ext in ("extension", "contradiction") and ext_conf == ext_conf and float(ext_conf) > 0.6:
        return str(ext)
    return str(row.get("predicted_intent", "background")).strip().lower()


def _load_edge_contexts(
    edges_path: Path = config.DATA_DIR / "edges_pred.parquet",
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Load (citing_id, cited_id) -> citation-context rows from edges_pred.parquet.

    Each row is {'sentence', 'intent', 'confidence'}. The key ("", cited_id) holds
    every sentence citing that paper. Returns {} if the file is missing.
    """
    key = str(edges_path)
    if key not in _EDGE_CONTEXT_CACHE:
        contexts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        if Path(edges_path).exists():
            import pandas as pd
            df = pd.read_parquet(edges_path)
            for row in df.to_dict("records"):
                sentence = str(row.get("context_sentence") or "").strip()
                if not sentence:
                    continue
                item = {
                    "sentence":   sentence,
                    "intent":     _edge_intent(row),
                    "confidence": float(row.get("intent_confidence") or 0.0),
                }
                contexts[(str(row["citing_id"]), str(row["cited_id"]))].append(item)
                contexts[("", str(row["cited_id"]))].append(item)   # all sentences citing this paper
        _EDGE_CONTEXT_CACHE[key] = dict(contexts)
    return _EDGE_CONTEXT_CACHE[key]


def _pick_context(
    pid: str,
    neighbour: str,
    intent: str,
    edge_contexts: dict[tuple[str, str], list[dict[str, Any]]],
) -> str:
    """Return the best citation-context sentence for the pid<->neighbour edge, or ''.

    Prefers a sentence whose intent equals `intent` (the strongest-edge intent),
    then the highest classifier confidence. Only sentences that cite pid are used.
    """
    if neighbour:
        # Only sentences written about pid (neighbour cites pid) describe pid itself
        rows = edge_contexts.get((neighbour, pid), [])
        if rows:
            same_intent = [r for r in rows if r["intent"] == intent]
            return max(same_intent or rows, key=lambda r: r["confidence"])["sentence"]
    # No edge inside the retrieved set: use the best sentence from any paper citing pid
    rows = edge_contexts.get(("", pid), [])
    return max(rows, key=lambda r: r["confidence"])["sentence"] if rows else ""


def _build_paper_records(
    ranked_ids: list[str],
    papers_lookup: dict[str, dict[str, Any]],
    G: nx.DiGraph,
    edge_contexts: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one structured record per paper.

    Record fields:
      paper_id, title, year, tldr, context, context_source, intent, neighbour
    context is a real citation-context sentence from the strongest edge to another
    retrieved paper when one exists (context_source='citation'); otherwise the TLDR
    ('tldr') or the first 200 characters of the abstract ('abstract').
    """
    retrieved_set = set(ranked_ids)
    edge_contexts = edge_contexts or {}
    records: dict[str, dict[str, Any]] = {}

    for pid in ranked_ids:
        meta = papers_lookup.get(pid, {})
        tldr = str(meta.get("tldr", "") or "").strip()
        year = meta.get("year", "n.d.")
        title = str(meta.get("title", "") or "(no title)").strip()
        abstract = str(meta.get("abstract", "") or "").strip()

        intent, neighbour = _dominant_intra_edge(pid, retrieved_set, G)

        # Prefer a real citation context; fall back to TLDR, then abstract[:200]
        context = _pick_context(pid, neighbour, intent, edge_contexts)
        if context:
            source = "citation"
        elif tldr:
            context, source = tldr, "tldr"
        else:
            context = abstract[:200] + "..." if len(abstract) > 200 else abstract
            source = "abstract"

        records[pid] = {
            "paper_id":       pid,
            "title":          title,
            "year":           year,
            "tldr":           tldr,
            "context":        context,
            "context_source": source,
            "intent":         intent,
            "neighbour":      neighbour,
        }
    return records


# ── Step 3: Intent-structured prompt ──────────────────────────────────────────

def _render_cluster_section(
    cluster_id: str | int,
    pids: list[str],
    records: dict[str, dict[str, Any]],
) -> str:
    """Render one cluster's paper list, sorted by intent order then year."""

    def sort_key(pid: str) -> tuple[int, int]:
        rec = records.get(pid, {})
        intent_rank = _INTENT_ORDER.get(rec.get("intent", "background"), 99)
        year_val = rec.get("year") or 0
        try:
            year_val = int(year_val)
        except (ValueError, TypeError):
            year_val = 0
        return (intent_rank, year_val)

    sorted_pids = sorted(pids, key=sort_key)

    cluster_label = (
        f"Cluster {cluster_id}" if cluster_id != "other" else "Unclustered papers"
    )
    lines = [f"[{cluster_label}]"]
    for pid in sorted_pids:
        rec = records[pid]
        intent = rec["intent"]
        role = _INTENT_ROLE.get(intent, intent)
        nbr = rec["neighbour"]
        nbr_note = f" (relative to [S2:{nbr}])" if nbr else ""
        ctx = rec["context"] or "(no summary available)"
        # Label real citation contexts differently from paper summaries
        ctx_label = "Citation context" if rec.get("context_source") == "citation" else "Summary"
        lines.append(
            f"  ID={pid}  ({rec['year']})  {rec['title']}\n"
            f"  Role: {role}{nbr_note}\n"
            f"  {ctx_label}: {ctx}"
        )
    return "\n\n".join(lines)


def _build_intent_prompt(
    query: str,
    clusters: dict[str | int, list[str]],
    records: dict[str, dict[str, Any]],
) -> str:
    """Build the full intent-structured prompt.

    Ordering of clusters: put the largest cluster first, then descending size.
    Within each cluster, background opens, then method/extension, then result,
    then contradiction.
    """
    # Sort clusters by descending size
    sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))

    sections: list[str] = []
    for cid, pids in sorted_clusters:
        sections.append(_render_cluster_section(cid, pids, records))

    papers_section = "\n\n".join(sections)

    prompt = (
        f"You are an expert academic writer.\n"
        f"Write a structured literature review (3-4 paragraphs) on the following topic:\n\n"
        f"TOPIC: {query}\n\n"
        f"The papers are grouped into thematic clusters based on citation community structure.\n"
        f"For each cluster, write at least one paragraph. Follow these rules strictly:\n"
        f"  1. Write EXACTLY ONE CLAIM PER SENTENCE.\n"
        f"  2. End EVERY sentence with [S2:<ID>] citing the paper that supports the claim.\n"
        f"     You may cite multiple papers: [S2:id1][S2:id2].\n"
        f"  3. Papers labelled 'provides background context' should open a paragraph.\n"
        f"  4. Papers labelled 'introduces or extends a method' or 'directly extends' should follow.\n"
        f"  5. Papers labelled 'compares or contrasts empirical results' should compare findings.\n"
        f"  6. Papers labelled 'contradicts or challenges findings' MUST appear in a sentence that\n"
        f"     begins with 'However,' or 'In contrast,' and explicitly names what is challenged.\n"
        f"  7. Do NOT invent citation IDs. Only use the IDs listed below.\n\n"
        f"--- PAPERS (grouped by cluster) ---\n\n"
        f"{papers_section}\n\n"
        f"--- END OF PAPERS ---\n\n"
        f"Write the review now:"
    )
    return prompt


# ── Main public interface ──────────────────────────────────────────────────────

def generate_review(
    query: str,
    ranked: list[dict[str, Any]],
    *,
    k: int = 10,
    model: str = DEFAULT_GROQ_MODEL,
    cache_dir: Path | str = CACHE_DIR,
    papers_path: Path | str = config.DATA_DIR / "papers.parquet",
    graph_path: Path | str = config.DATA_DIR / "graph.gpickle",
    edges_pred_path: Path | str = config.DATA_DIR / "edges_pred.parquet",
    verbose: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Generate an intent-aware literature review from a ranked list of papers.

    This is the shared-contract entry point for Step 14b.

    Args:
        query:       Natural-language query.
        ranked:      Retrieval-contract dicts (paper_id, score, rank, ...).
                     Only the first ``k`` entries are used.
        k:           Number of top-ranked papers to include.
        model:       Groq model identifier.
        cache_dir:   Directory for caching LLM responses.
        papers_path: Path to papers.parquet.
        graph_path:  Path to graph.gpickle (for community IDs and edge intents).
        verbose:     Print progress to stdout.
        use_cache:   If False, skip the disk cache (no read, no write).

    Returns:
        dict with keys:
          'text'           -- Full generated review text.
          'claims'         -- list of {'sentence': str, 'paper_ids': list[str]}.
          'n_invalid_tags' -- Count of [S2:...] tags not in the ranked list.
          'model'          -- Groq model used.
          'query'          -- Original query.
          'paper_ids_used' -- Ordered list of paper IDs passed to the LLM.
          'from_cache'     -- True if the response came from disk cache.
          'clusters'       -- dict mapping community_id -> list[paper_id].
          'paper_records'  -- paper_id -> {title, year, tldr, context, context_source,
                              intent, neighbour}; traces a claim's [S2:id] to its
                              citation context and intent label.

    Raises:
        EnvironmentError:  If GROQ_API_KEY is not set.
        FileNotFoundError: If papers.parquet is missing.
    """
    api_key = os.environ.get(GROQ_API_KEY_ENV, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"GROQ_API_KEY is not set. Add it to your .env file.\n"
            f"  Example: {GROQ_API_KEY_ENV}=gsk_..."
        )

    cache_dir = Path(cache_dir)
    papers_path = Path(papers_path)
    graph_path = Path(graph_path)

    # Resolve top-k
    top_k_entries = ranked[:k]
    ranked_ids_ordered = [str(r["paper_id"]) for r in top_k_entries]
    ranked_ids_set = set(ranked_ids_ordered)

    if not papers_path.exists():
        raise FileNotFoundError(f"papers.parquet not found at {papers_path}")

    # ── Load graph ────────────────────────────────────────────────────────────
    try:
        G = _get_graph(graph_path)
    except FileNotFoundError:
        if verbose:
            print("[generate] WARNING: graph.gpickle not found; all papers go to 'other' cluster.")
        G = nx.DiGraph()

    # ── Load paper metadata ───────────────────────────────────────────────────
    papers_lookup = _load_papers_lookup(papers_path)

    # ── Step 1: cluster by Louvain community ──────────────────────────────────
    clusters = _cluster_by_community(ranked_ids_ordered, G)
    if verbose:
        n_clusters = len(clusters)
        print(f"[generate] {len(ranked_ids_ordered)} papers in {n_clusters} cluster(s): "
              + ", ".join(f"{cid}({len(pids)})" for cid, pids in clusters.items()))

    # ── Step 2: build structured records ─────────────────────────────────────
    records = _build_paper_records(
        ranked_ids_ordered, papers_lookup, G,
        edge_contexts=_load_edge_contexts(edges_pred_path),
    )

    # ── Step 3: build intent-structured prompt ────────────────────────────────
    prompt = _build_intent_prompt(query, clusters, records)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_key = _cache_key(prompt)
    cached_text = _load_cache(cache_dir, cache_key) if use_cache else None
    from_cache = cached_text is not None

    if from_cache:
        if verbose:
            print(f"[generate] Cache HIT ({cache_key[:12]}...)")
        review_text = cached_text
    else:
        if verbose:
            print(f"[generate] Cache MISS -- calling Groq (model={model})...")
        review_text = _call_groq(prompt, model=model, api_key=api_key)
        if use_cache:
            _save_cache(cache_dir, cache_key, review_text)
            if verbose:
                print(f"[generate] Cached -> {cache_dir / (cache_key + '.json')}")

    # ── Post-process ──────────────────────────────────────────────────────────
    claims, n_invalid = _extract_claims(review_text, ranked_ids_set)

    if n_invalid > 0:
        print(
            f"[generate] WARNING: {n_invalid} citation tag(s) reference paper IDs "
            f"not in the ranked list (hallucinated or out-of-range)."
        )

    return {
        "text":            review_text,
        "claims":          claims,
        "n_invalid_tags":  n_invalid,
        "model":           model,
        "query":           query,
        "paper_ids_used":  ranked_ids_ordered,
        "from_cache":      from_cache,
        "clusters":        {str(k_): v for k_, v in clusters.items()},
        "paper_records":   records,
    }


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    # Force UTF-8 output so LLM Unicode chars don't crash cp1252 terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Step 14b: Citation-aware literature review generator (Groq LLM)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Graph neural networks for collaborative filtering in recommender systems",
        help="Query string to generate a review for.",
    )
    parser.add_argument(
        "-k", type=int, default=10,
        help="Number of top-k papers to include in the prompt (default 10).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_GROQ_MODEL,
        help=f"Groq model identifier (default: {DEFAULT_GROQ_MODEL}).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore the disk cache and always call the API.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress messages.",
    )
    args = parser.parse_args()

    print(f"Query: {args.query!r}")
    print("Running retrieval (stage1 + graph re-ranking)...")

    from retrieval import retrieve
    ranked = retrieve(args.query, k=args.k, surface_contradictions=False)

    if not ranked:
        print("No results returned by retrieve(). Cannot generate a review.", file=sys.stderr)
        sys.exit(1)

    result = generate_review(
        args.query,
        ranked,
        k=args.k,
        model=args.model,
        use_cache=not args.no_cache,
        verbose=True,
    )

    # ── Pretty-print ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATED REVIEW (citation-aware)")
    print("=" * 80)
    print(result["text"])
    print("=" * 80)

    print(f"\nClusters:")
    for cid, pids in result["clusters"].items():
        print(f"  [{cid}] ({len(pids)} papers): {', '.join(p[:12] for p in pids)}")

    n_sent  = len(result["claims"])
    n_cited = sum(1 for c in result["claims"] if c["paper_ids"])
    n_valid = sum(len(c["paper_ids"]) for c in result["claims"])
    n_inv   = result["n_invalid_tags"]

    print(f"\nTotal sentences:     {n_sent}")
    print(f"Sentences with cite: {n_cited}")
    print(f"Total valid tags:    {n_valid}")
    if n_inv > 0:
        print(f"Invalid tags:        {n_inv}  <- hallucinated IDs!")
    else:
        print("Invalid tags:        0  (no hallucinated IDs)")
    print(f"From cache:          {result['from_cache']}")
    print(f"Model:               {result['model']}")

    print("\n--- PER-SENTENCE CLAIMS ---")
    for i, claim in enumerate(result["claims"], 1):
        if claim["paper_ids"]:
            print(f"[{i}] {claim['sentence']}")
            print(f"     -> cites: {claim['paper_ids']}")
            for pid in claim["paper_ids"]:
                rec = result["paper_records"].get(pid)
                if rec:
                    print(f"        [{pid[:8]}] intent={rec['intent']} "
                          f"({rec['context_source']}): {rec['context'][:120]}")


if __name__ == "__main__":
    main()
