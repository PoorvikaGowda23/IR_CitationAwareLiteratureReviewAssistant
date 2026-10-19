"""
evaluate.py
-----------
Two responsibilities:
  1. build_ground_truth() -- reads data/queries.csv + data/papers.parquet,
     filters relevant IDs to corpus members, drops thin queries, performs a
     stratified val/test split, and writes data/ground_truth.json.
  2. (Step 7+) compute nDCG@10, MAP, and P@10 against that ground truth.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from config import ALLOW_SURVEY_FALLBACK, DATA_DIR, MIN_RELEVANT, SEED, YEAR_TOLERANCE
from schema import (
    GT_QUERY_ID,
    GT_QUERY_TEXT,
    GT_RELEVANT_PAPER_IDS,
    GT_SPLIT,
    PAPER_ID,
    load_papers,
    load_ground_truth,
    save_ground_truth,
)

# -- Constants -----------------------------------------------------------------
QUERIES_CSV       = DATA_DIR / "queries.csv"
SURVEY_REFS_JSON  = DATA_DIR / "survey_refs.json"
PAPERS_PARQUET    = DATA_DIR / "papers.parquet"
EDGES_PARQUET     = DATA_DIR / "edges.parquet"
GROUND_TRUTH_JSON = DATA_DIR / "ground_truth.json"

# Fraction of queries assigned to the validation split (rest go to test).
VAL_FRACTION: float = 0.30


# -- Ground-truth validator ----------------------------------------------------

def validate_ground_truth(
    queries: list[dict],
    survey_refs: dict[str, list[str]],
    papers_df: pd.DataFrame,
    edges_parquet: Path = EDGES_PARQUET,
    year_tolerance: int = YEAR_TOLERANCE,
) -> None:
    """Validate ground-truth queries for survey consistency, temporal validity, and graph presence.

    Requirements:
      1. Every relevant ID must belong to survey_refs[survey_id] (hard error if violated).
      2. If a relevant paper's year > survey year + year_tolerance, raise ValueError.
         If later than survey year but within tolerance, print a per-query WARNING and
         summarize totals per survey at the end.
      3. Report count of relevant IDs appearing in 5 or more queries (prints warning if > 0).
      4. Report share of relevant IDs that have >= 1 edge in edges.parquet (prints warning if < 1.0).

    Raises:
        ValueError: if any ID is outside survey_refs[survey_id] or exceeds survey year + tolerance.
    """
    if not queries:
        return

    papers_indexed = (
        papers_df.set_index(PAPER_ID)
        if PAPER_ID in papers_df.columns
        else papers_df
    )

    survey_tolerated = defaultdict(set)

    for q in queries:
        qid = q.get("query_id", "")
        sid = q.get("survey_id", "")
        rel_ids = q.get("relevant_ids", q.get(GT_RELEVANT_PAPER_IDS, []))

        allowed_refs = set(survey_refs.get(sid, []))
        survey_year = None
        if sid in papers_indexed.index:
            s_val = papers_indexed.loc[sid, "year"]
            if pd.notna(s_val):
                survey_year = float(s_val)

        query_tolerated_pids = []
        query_gaps = []

        for pid in rel_ids:
            # Check 1: in survey_refs[survey_id] (hard error)
            if pid not in allowed_refs:
                raise ValueError(
                    f"Ground-truth validation failed for query '{qid}': "
                    f"relevant paper '{pid}' is NOT cited in survey '{sid}'."
                )

            # Check 2: year check with tolerance
            if pid in papers_indexed.index and survey_year is not None:
                p_val = papers_indexed.loc[pid, "year"]
                if pd.notna(p_val):
                    p_year = float(p_val)
                    if p_year > survey_year + year_tolerance:
                        raise ValueError(
                            f"Ground-truth validation failed for query '{qid}': "
                            f"relevant paper '{pid}' was published in {int(p_year)}, "
                            f"which is later than survey '{sid}' ({int(survey_year)}) + tolerance ({year_tolerance})."
                        )
                    elif p_year > survey_year:
                        gap = int(p_year - survey_year)
                        query_tolerated_pids.append(pid)
                        query_gaps.append(gap)
                        survey_tolerated[sid].add(pid)

        # After validating each query, print one WARNING line if any papers are later than survey year
        if query_tolerated_pids:
            max_gap = max(query_gaps)
            print(
                f"WARNING: [{qid}] {len(query_tolerated_pids)} relevant paper(s) published "
                f"later than survey year ({int(survey_year)}), max gap: +{max_gap} year(s) (within tolerance)."
            )

    # Print total number of papers per survey that were later than survey year but within tolerance
    unique_surveys = list(dict.fromkeys(q.get("survey_id", "") for q in queries if q.get("survey_id")))
    print(f"\n[Validation] Total papers per survey later than survey year (within tolerance <= +{year_tolerance}y):")
    for sid in unique_surveys:
        print(f"  survey_id='{sid}': {len(survey_tolerated[sid])} paper(s)")

    # Check 3: ID frequency across queries
    all_rel_ids = [
        pid
        for q in queries
        for pid in q.get("relevant_ids", q.get(GT_RELEVANT_PAPER_IDS, []))
    ]
    id_counts = Counter(all_rel_ids)
    frequent_ids = {pid: count for pid, count in id_counts.items() if count >= 5}
    print(f"\n[Validation] Relevant IDs appearing in >= 5 queries: {len(frequent_ids)}")
    if frequent_ids:
        print(f"WARNING: {len(frequent_ids)} relevant ID(s) appear in 5 or more queries: {frequent_ids}")

    # Check 4: Share of relevant IDs in edges.parquet
    edges_path = Path(edges_parquet)
    if edges_path.exists():
        edges_df = pd.read_parquet(edges_path)
        edge_nodes = set(edges_df["citing_id"].astype(str)).union(set(edges_df["cited_id"].astype(str)))
        unique_rel_ids = set(all_rel_ids)
        if unique_rel_ids:
            in_edges = {pid for pid in unique_rel_ids if pid in edge_nodes}
            share = len(in_edges) / len(unique_rel_ids)
            print(f"[Validation] Share of relevant IDs with >= 1 edge in {edges_path.name}: {share:.1%} ({len(in_edges)}/{len(unique_rel_ids)})")
            if share < 1.0:
                print(f"WARNING: {len(unique_rel_ids) - len(in_edges)} relevant ID(s) have no citation edges in {edges_path.name}.")
        else:
            print(f"[Validation] No relevant IDs to check in {edges_path.name}.")
    else:
        print(f"WARNING: {edges_path} not found -- skipping citation edge check.")


# -- Ground-truth builder ------------------------------------------------------

def build_ground_truth(
    queries_csv: Path = QUERIES_CSV,
    survey_refs_json: Path = SURVEY_REFS_JSON,
    papers_parquet: Path = PAPERS_PARQUET,
    output_json: Path = GROUND_TRUTH_JSON,
    min_relevant: int = MIN_RELEVANT,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> list[dict]:
    """Build and save data/ground_truth.json from a hand-written queries CSV.

    Pipeline
    --------
    1. Load data/queries.csv (columns: query_id, query_text, survey_id,
       relevant_ids [optional, semicolon-separated S2 paper IDs]).
    2. If relevant_ids is empty for a row, fall back to that survey'\''s full
       reference list from data/survey_refs.json.
    3. Intersect each relevant set with the papers in data/papers.parquet
       (corpus filter).  Print per-query coverage.
    4. Drop any query whose filtered relevant set has fewer than ``min_relevant``
       papers; print which queries are dropped and why.
    5. Assign a '\''val'\'' or '\''test'\'' split with StratifiedShuffleSplit (stratified
       by survey_id, fixed seed).
    6. Write data/ground_truth.json and print a summary.

    Args:
        queries_csv:      Path to the hand-written queries CSV.
        survey_refs_json: Path to survey_refs.json produced by scrape.py.
        papers_parquet:   Path to papers.parquet produced by scrape.py.
        output_json:      Destination for ground_truth.json.
        min_relevant:     Drop queries with fewer corpus-relevant IDs than this.
        val_fraction:     Approximate fraction of queries assigned to '\''val'\''.
        seed:             Random seed for reproducible splits.

    Returns:
        List of ground-truth records written to output_json.

    Raises:
        FileNotFoundError: if queries_csv or papers_parquet are missing.
        ValueError:        if queries_csv is missing required columns.
    """

    # -- 1. Load inputs --------------------------------------------------------
    if not queries_csv.exists():
        raise FileNotFoundError(
            f"Queries CSV not found: {queries_csv}\n"
            "Please create it with columns: query_id, query_text, survey_id, "
            "relevant_ids (semicolon-separated, may be empty)."
        )

    queries_df = pd.read_csv(queries_csv, dtype=str).fillna("")

    required_cols = {"query_id", "query_text", "survey_id"}
    missing_cols = required_cols - set(queries_df.columns)
    if missing_cols:
        raise ValueError(
            f"queries.csv is missing required columns: {sorted(missing_cols)}\n"
            f"Found columns: {list(queries_df.columns)}"
        )

    if "relevant_ids" not in queries_df.columns:
        queries_df["relevant_ids"] = ""

    if "query_type" not in queries_df.columns:
        queries_df["query_type"] = "section"

    # Load the corpus -- we need the set of paper IDs that actually exist.
    papers_df = load_papers(papers_parquet)
    corpus_ids: set[str] = set(papers_df[PAPER_ID].astype(str))

    # Load survey_refs.json for fallback (survey_id -> list[paper_id]).
    survey_refs: dict[str, list[str]] = {}
    if survey_refs_json.exists():
        with open(survey_refs_json, "r", encoding="utf-8") as f:
            survey_refs = json.load(f)
    else:
        print(f"WARNING: {survey_refs_json} not found -- fallback to survey refs unavailable.")

    # -- 2-3. Resolve and filter relevant IDs ---------------------------------
    print("\n" + "=" * 68)
    print("  Per-query relevant-ID coverage")
    print("=" * 68)
    print(f"  {'query_id':<30} {'in_file':>7}  {'in_corpus':>9}  status")
    print("-" * 68)

    resolved_rows: list[dict] = []

    for _, row in queries_df.iterrows():
        qid        = str(row["query_id"]).strip()
        query_text = str(row["query_text"]).strip()
        survey_id  = str(row["survey_id"]).strip()
        query_type = str(row.get("query_type", "section")).strip().lower()
        if not query_type:
            query_type = "section"
        raw_ids    = str(row["relevant_ids"]).strip()

        # Parse the semicolon-separated ID list; optionally fall back to survey refs.
        fallback_used = False
        if raw_ids:
            file_ids = [s.strip() for s in raw_ids.split(";") if s.strip()]
        elif query_type == "survey":
            # For 'survey' rows, relevant_ids may be left empty and the whole survey
            # reference list is used (this is the only allowed fallback).
            file_ids = survey_refs.get(survey_id, [])
            fallback_used = True
            print(
                f"INFO: [{qid}] (query_type='survey') relevant_ids is empty -- "
                f"using full survey reference list ({len(file_ids)} refs for survey_id='{survey_id}')."
            )
        elif ALLOW_SURVEY_FALLBACK:
            file_ids = survey_refs.get(survey_id, [])
            fallback_used = True
            print(
                f"WARNING: [{qid}] relevant_ids is empty -- "
                f"falling back to full survey reference list "
                f"({len(file_ids)} refs for survey_id='{survey_id}')."
            )
        else:
            file_ids = []

        # Intersect with corpus.
        corpus_ids_for_query = [pid for pid in file_ids if pid in corpus_ids]

        # Remove duplicates while preserving order.
        seen: set[str] = set()
        filtered: list[str] = []
        for pid in corpus_ids_for_query:
            if pid not in seen:
                seen.add(pid)
                filtered.append(pid)

        n_file   = len(file_ids)
        n_corpus = len(filtered)

        resolved_rows.append({
            "query_id":     qid,
            "query_text":   query_text,
            "survey_id":    survey_id,
            "query_type":   query_type,
            "relevant_ids": filtered,
            "n_file":       n_file,
            "n_corpus":     n_corpus,
        })

        if n_corpus >= min_relevant:
            status = "OK"
        elif not raw_ids and query_type != "survey" and not ALLOW_SURVEY_FALLBACK:
            status = "DROP (no ids, fallback disabled)"
        else:
            status = f"DROP (<{min_relevant})"
        # Safe-encode for Windows terminals that may not support all Unicode.
        safe_qid = qid.encode("ascii", errors="replace").decode("ascii")
        print(f"  {safe_qid:<30} {n_file:>7}  {n_corpus:>9}  {status}")

    print("=" * 68)

    # -- 4. Drop thin queries -------------------------------------------------
    kept_rows    = [r for r in resolved_rows if r["n_corpus"] >= min_relevant]
    dropped_rows = [r for r in resolved_rows if r["n_corpus"] <  min_relevant]

    if dropped_rows:
        print(f"\nDROPPED {len(dropped_rows)} query/queries:")
        for r in dropped_rows:
            safe_qid = r["query_id"].encode("ascii", errors="replace").decode("ascii")
            if r["n_file"] == 0 and r.get("query_type") != "survey" and not ALLOW_SURVEY_FALLBACK:
                reason = "relevant_ids is empty and fallback disabled for section queries"
            else:
                reason = (
                    f"only {r['n_corpus']} corpus-relevant IDs "
                    f"(had {r['n_file']} in file, need >= {min_relevant})"
                )
            print(f"  [{safe_qid}] {reason}")
    else:
        print(f"\nNo queries dropped -- all have >= {min_relevant} corpus-relevant IDs.")

    if not kept_rows:
        raise ValueError(
            "All queries were dropped. Either add more relevant IDs to queries.csv "
            "or lower MIN_RELEVANT in evaluate.py."
        )

    # -- 4b. Validate ground truth integrity ----------------------------------
    validate_ground_truth(
        queries=kept_rows,
        survey_refs=survey_refs,
        papers_df=papers_df,
        edges_parquet=EDGES_PARQUET,
    )

    # -- 5. Stratified val / test split ---------------------------------------
    kept_df = pd.DataFrame(kept_rows)

    # StratifiedShuffleSplit requires at least 2 samples per class.
    # If a survey has only 1 query it can't be stratified -- fall back gracefully.
    survey_counts     = kept_df["survey_id"].value_counts()
    singleton_surveys = survey_counts[survey_counts < 2].index.tolist()

    if singleton_surveys:
        print(
            f"\nWARNING: survey_id(s) {singleton_surveys} have only 1 query each; "
            "they will be assigned to 'test' (not enough samples to stratify)."
        )
        mask_singleton = kept_df["survey_id"].isin(singleton_surveys)
        singleton_df   = kept_df[mask_singleton].copy()
        stratify_df    = kept_df[~mask_singleton].copy()
        singleton_df["split"] = "test"
    else:
        singleton_df = pd.DataFrame()
        stratify_df  = kept_df.copy()

    # Apply stratified split to the non-singleton portion.
    if len(stratify_df) > 0:
        n_classes = stratify_df["survey_id"].nunique()
        # StratifiedShuffleSplit requires at least 1 sample per class in both splits (train and test).
        if len(stratify_df) >= 2 * n_classes:
            val_count = max(n_classes, int(round(val_fraction * len(stratify_df))))
            val_count = min(val_count, len(stratify_df) - n_classes)
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=val_count,
                random_state=seed,
            )
            val_idx, test_idx = next(sss.split(stratify_df, stratify_df["survey_id"]))
        else:
            # If dataset is too small to have n_classes in both splits, pick 1 per class for val
            val_indices = (
                stratify_df.groupby("survey_id", group_keys=False)
                .apply(lambda g: g.sample(n=1, random_state=seed))
                .index
            )
            val_idx = [stratify_df.index.get_loc(i) for i in val_indices]

        stratify_df = stratify_df.copy()
        stratify_df["split"] = "test"
        stratify_df.iloc[val_idx, stratify_df.columns.get_loc("split")] = "val"
    else:
        stratify_df = pd.DataFrame()

    # Recombine.
    split_df = pd.concat([stratify_df, singleton_df], ignore_index=True)

    # -- 6. Assemble records and save ----------------------------------------
    records: list[dict] = []
    for _, row in split_df.iterrows():
        records.append({
            GT_QUERY_ID:            row["query_id"],
            GT_QUERY_TEXT:          row["query_text"],
            GT_SPLIT:               row["split"],
            GT_RELEVANT_PAPER_IDS:  row["relevant_ids"],
            "query_type":           row.get("query_type", "section"),
        })

    save_ground_truth(records, output_json)

    # -- 7. Summary -----------------------------------------------------------
    def _rel_stats(recs: list[dict]) -> tuple[float, int]:
        """Return (average, minimum) relevant-ID count for a list of records."""
        counts = [len(r[GT_RELEVANT_PAPER_IDS]) for r in recs]
        if not counts:
            return 0.0, 0
        return sum(counts) / len(counts), min(counts)

    print("\n" + "=" * 68)
    print("  Ground-truth split summary")
    print("=" * 68)
    print(f"  Total queries kept : {len(records)}")
    print(f"  Total dropped      : {len(dropped_rows)}")

    query_types = sorted(list({r.get("query_type", "section") for r in records}))
    for qt in query_types:
        qt_recs = [r for r in records if r.get("query_type", "section") == qt]
        val_recs  = [r for r in qt_recs if r[GT_SPLIT] == "val"]
        test_recs = [r for r in qt_recs if r[GT_SPLIT] == "test"]
        val_avg,  val_min  = _rel_stats(val_recs)
        test_avg, test_min = _rel_stats(test_recs)
        print(f"\n  [Query Type: {qt}] ({len(qt_recs)} kept)")
        print(f"    Val  queries     : {len(val_recs):<5}  avg relevant = {val_avg:.1f},  min = {val_min}")
        print(f"    Test queries     : {len(test_recs):<5}  avg relevant = {test_avg:.1f},  min = {test_min}")

    print(f"\n  Saved -> {output_json}")
    print("=" * 68)

    return records


# ── Step 7: Retrieval Evaluation Metrics (from scratch, binary relevance) ─────

def precision_at_k(ranked_ids: list[str], relevant: set[str] | list[str], k: int) -> float:
    """Compute Precision@k for binary relevance.

    Args:
        ranked_ids: Ordered list of retrieved paper IDs.
        relevant:   Set or list of ground-truth relevant paper IDs.
        k:          Rank cutoff.

    Returns:
        Precision@k as a float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_set)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant: set[str] | list[str], k: int) -> float:
    """Compute Recall@k for binary relevance.

    Args:
        ranked_ids: Ordered list of retrieved paper IDs.
        relevant:   Set or list of ground-truth relevant paper IDs.
        k:          Rank cutoff.

    Returns:
        Recall@k as a float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_set)
    return hits / len(relevant_set)


def average_precision(ranked_ids: list[str], relevant: set[str] | list[str]) -> float:
    """Compute Average Precision (AP) for binary relevance across all ranked items.

    Args:
        ranked_ids: Ordered list of retrieved paper IDs.
        relevant:   Set or list of ground-truth relevant paper IDs.

    Returns:
        Average Precision as a float in [0.0, 1.0].
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant_set)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str] | list[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@k) for binary relevance.

    Uses standard log2(rank + 1) discounting:
      DCG@k  = sum_{i=1..k} (rel_i / log2(i + 1))
      IDCG@k = sum_{i=1..min(k, |relevant|)} (1 / log2(i + 1))

    Args:
        ranked_ids: Ordered list of retrieved paper IDs.
        relevant:   Set or list of ground-truth relevant paper IDs.
        k:          Rank cutoff.

    Returns:
        nDCG@k as a float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ── Statistical Significance Helpers ──────────────────────────────────────────

def bootstrap_ci(
    per_query_scores: list[float],
    n: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    """Compute a 95% bootstrap confidence interval for the mean metric score.

    Args:
        per_query_scores: List of metric values (one per query).
        n:                Number of bootstrap resamples.
        seed:             Random seed.

    Returns:
        (lower_bound, upper_bound) tuple of 95% confidence interval.
    """
    if not per_query_scores:
        return (0.0, 0.0)
    scores_arr = np.array(per_query_scores, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = [
        rng.choice(scores_arr, size=len(scores_arr), replace=True).mean()
        for _ in range(n)
    ]
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return (low, high)


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n: int = 1000,
    seed: int = SEED,
) -> float:
    """Compute empirical probability that method A outperforms method B: P(A > B).

    Uses paired bootstrap resampling over per-query differences.

    Args:
        scores_a: Per-query scores from method A.
        scores_b: Per-query scores from method B.
        n:        Number of bootstrap iterations.
        seed:     Random seed.

    Returns:
        Probability in [0.0, 1.0] that A beats B.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length ({len(scores_a)} vs {len(scores_b)})"
        )
    if not scores_a:
        return 0.5

    diffs = np.array(scores_a, dtype=float) - np.array(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    wins = 0.0
    for _ in range(n):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        m = sample.mean()
        if m > 0:
            wins += 1.0
        elif m == 0:
            wins += 0.5
    return float(wins / n)


# ── Unified Evaluator ─────────────────────────────────────────────────────────

def evaluate(
    retrieve_fn,
    queries: Any = None,
    ground_truth: Any = GROUND_TRUTH_JSON,
    split: str = "val",
    k: int = 50,
) -> dict[str, Any]:
    """Evaluate a retrieval function against ground-truth queries.

    Args:
        retrieve_fn:  Callable (query_text, k=...) returning list[dict] or list[str].
        queries:      Optional queries source (defaults to ground_truth records).
        ground_truth: Path to ground_truth.json or list of ground truth records.
        split:        Split to evaluate on ('val' or 'test').
        k:            Number of candidates to fetch (default: 50).

    Returns:
        dict containing:
          - 'ndcg@10': mean nDCG@10
          - 'map': mean Average Precision
          - 'p@10': mean Precision@10
          - 'recall@50': mean Recall@50 (if retrieve_fn returns up to 50 results)
          - 'per_query': mapping query_id -> metric dict
          - 'scores': mapping metric_name -> list of per-query floats
          - 'split': split name
          - 'num_queries': count of queries evaluated
    """
    if isinstance(ground_truth, (str, Path)):
        gt_records = load_ground_truth(ground_truth)
    elif isinstance(ground_truth, list):
        gt_records = ground_truth
    else:
        gt_records = load_ground_truth(GROUND_TRUTH_JSON)

    split_recs = [r for r in gt_records if r.get(GT_SPLIT) == split]
    if not split_recs:
        raise ValueError(f"No records found for split '{split}' in ground truth.")

    per_query: dict[str, dict[str, float]] = {}
    metric_lists: dict[str, list[float]] = {
        "ndcg@10": [],
        "map": [],
        "p@10": [],
        "recall@50": [],
    }

    has_50_support = True

    for rec in split_recs:
        qid = rec[GT_QUERY_ID]
        qtext = rec[GT_QUERY_TEXT]
        relevant = set(rec[GT_RELEVANT_PAPER_IDS])

        results = retrieve_fn(qtext, k=k)
        if results and isinstance(results[0], dict):
            ranked_ids = [r[PAPER_ID] for r in results]
        else:
            ranked_ids = list(results)

        q_ndcg10 = ndcg_at_k(ranked_ids, relevant, k=10)
        q_map    = average_precision(ranked_ids, relevant)
        q_p10    = precision_at_k(ranked_ids, relevant, k=10)

        q_dict = {
            "ndcg@10": q_ndcg10,
            "map": q_map,
            "p@10": q_p10,
        }
        metric_lists["ndcg@10"].append(q_ndcg10)
        metric_lists["map"].append(q_map)
        metric_lists["p@10"].append(q_p10)

        if len(ranked_ids) >= 50 or k >= 50:
            q_r50 = recall_at_k(ranked_ids, relevant, k=50)
            q_dict["recall@50"] = q_r50
            metric_lists["recall@50"].append(q_r50)
        else:
            has_50_support = False

        per_query[qid] = q_dict

    out: dict[str, Any] = {
        "ndcg@10": float(np.mean(metric_lists["ndcg@10"])),
        "map": float(np.mean(metric_lists["map"])),
        "p@10": float(np.mean(metric_lists["p@10"])),
        "per_query": per_query,
        "scores": metric_lists,
        "split": split,
        "num_queries": len(split_recs),
    }

    if has_50_support and metric_lists["recall@50"]:
        out["recall@50"] = float(np.mean(metric_lists["recall@50"]))

    return out


# -- CLI entry-point -----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ground Truth Builder & Retrieval Evaluator."
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Rebuild data/ground_truth.json from data/queries.csv."
    )
    parser.add_argument(
        "--queries", type=Path, default=QUERIES_CSV,
        help=f"Path to queries CSV (default: {QUERIES_CSV})"
    )
    parser.add_argument(
        "--survey-refs", type=Path, default=SURVEY_REFS_JSON,
        help=f"Path to survey_refs.json (default: {SURVEY_REFS_JSON})"
    )
    parser.add_argument(
        "--papers", type=Path, default=PAPERS_PARQUET,
        help=f"Path to papers.parquet (default: {PAPERS_PARQUET})"
    )
    parser.add_argument(
        "--out", type=Path, default=GROUND_TRUTH_JSON,
        help=f"Output path for ground_truth.json (default: {GROUND_TRUTH_JSON})"
    )
    parser.add_argument(
        "--min-relevant", type=int, default=MIN_RELEVANT,
        help=f"Minimum corpus-present relevant IDs to keep a query (default: {MIN_RELEVANT})"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help=f"Fraction of queries assigned to val split (default: {VAL_FRACTION})"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Split to evaluate on (default: val)."
    )
    args = parser.parse_args()

    if args.build:
        build_ground_truth(
            queries_csv      = args.queries,
            survey_refs_json = args.survey_refs,
            papers_parquet   = args.papers,
            output_json      = args.out,
            min_relevant     = args.min_relevant,
            val_fraction     = args.val_fraction,
        )
    else:
        # Default behavior: evaluate retrieval baselines on the specified split
        from retrieval_baseline import retrieve_baseline, retrieve_bm25

        print(f"\n" + "=" * 78)
        print(f"  EVALUATION RESULTS: Dense Baseline vs. BM25 (Split: '{args.split}')")
        print("=" * 78)

        res_dense = evaluate(retrieve_baseline, ground_truth=args.out, split=args.split)
        res_bm25  = evaluate(retrieve_bm25, ground_truth=args.out, split=args.split)

        metrics = ["ndcg@10", "map", "p@10"]
        if "recall@50" in res_dense and "recall@50" in res_bm25:
            metrics.append("recall@50")

        # Table Header
        print(f"\n{'Metric':<12} | {'Dense (SPECTER2)':<22} | {'Lexical (BM25)':<22} | {'P(Dense > BM25)':<15}")
        print("-" * 78)

        for m in metrics:
            dense_mean = res_dense[m]
            dense_ci   = bootstrap_ci(res_dense["scores"][m])
            bm25_mean  = res_bm25[m]
            bm25_ci    = bootstrap_ci(res_bm25["scores"][m])
            p_beat     = paired_bootstrap(res_dense["scores"][m], res_bm25["scores"][m])

            dense_str = f"{dense_mean:.4f} [{dense_ci[0]:.3f}, {dense_ci[1]:.3f}]"
            bm25_str  = f"{bm25_mean:.4f} [{bm25_ci[0]:.3f}, {bm25_ci[1]:.3f}]"
            print(f"{m:<12} | {dense_str:<22} | {bm25_str:<22} | {p_beat:<15.3f}")

        print("=" * 78)

        print(f"\nPer-Query Breakdown ({args.split} split, {res_dense['num_queries']} queries):")
        print(f"{'Query ID':<16} | {'Metric':<10} | {'Dense':<8} | {'BM25':<8}")
        print("-" * 50)
        for qid in sorted(res_dense["per_query"].keys()):
            for m in metrics:
                d_val = res_dense["per_query"][qid].get(m, 0.0)
                b_val = res_bm25["per_query"][qid].get(m, 0.0)
                print(f"{qid:<16} | {m:<10} | {d_val:<8.4f} | {b_val:<8.4f}")
            print("-" * 50)

