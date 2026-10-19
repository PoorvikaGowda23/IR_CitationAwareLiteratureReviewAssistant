"""
section_refs.py
---------------
Utility to inspect and assign ground-truth reference papers for section-level queries.
Ensures that all assigned paper IDs:
  1. Strictly belong to the survey's reference list (data/survey_refs.json).
  2. Have publication years within tolerance of the survey year.
  3. Are present in the local corpus (data/papers.parquet).

Usage:
  python section_refs.py --inspect
  python section_refs.py --inspect --query s1_gcf
  python section_refs.py --apply
"""

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from config import DATA_DIR, MIN_RELEVANT, YEAR_TOLERANCE
from schema import ABSTRACT, IS_SURVEY, PAPER_ID, TITLE, YEAR, load_papers

QUERIES_CSV = DATA_DIR / "queries.csv"
SURVEY_REFS_JSON = DATA_DIR / "survey_refs.json"
PAPERS_PARQUET = DATA_DIR / "papers.parquet"

# Specific topic terms to match high-precision section references within each survey
TOPIC_KEYWORDS = {
    "s1_gcf": [
        "collaborative filtering", "collaborative", "user-item", "bipartite",
        "ngcf", "lightgcn", "gc-mc", "spectral collaborative"
    ],
    "s1_lightgcn": [
        "lightgcn", "simplif", "linear graph convolution", "sgc", "layer combination"
    ],
    "s1_session": [
        "session", "sequential", "next-item", "sr-gnn", "item graph", "order"
    ],
    "s2_kg": [
        "knowledge graph", "knowledge-aware", "entity", "kgat", "ripplenet", "cbrec"
    ],
    "s2_contrastive": [
        "contrastive", "self-supervised", "data augmentation", "mutual information", "ssl"
    ],
    "s3_hypergraph": [
        "hypergraph", "hyperedge", "group recommendation", "high-order relation"
    ],
    "s3_dynamic": [
        "dynamic", "temporal", "time-aware", "sequential", "evolving", "time"
    ],
}


def inspect_section_candidates(
    query_id: str | None = None,
    queries_csv: Path = QUERIES_CSV,
    survey_refs_json: Path = SURVEY_REFS_JSON,
    papers_parquet: Path = PAPERS_PARQUET,
    min_relevant: int = MIN_RELEVANT,
    year_tolerance: int = YEAR_TOLERANCE,
) -> dict[str, list[str]]:
    """Inspect survey references matching each section query."""
    if not queries_csv.exists() or not survey_refs_json.exists() or not papers_parquet.exists():
        print("Required data files not found. Ensure papers.parquet and survey_refs.json exist.")
        return {}

    papers_df = load_papers(papers_parquet).set_index(PAPER_ID)
    queries_df = pd.read_csv(queries_csv, dtype=str).fillna("")
    survey_refs = json.load(open(survey_refs_json, "r", encoding="utf-8"))

    section_queries = queries_df[queries_df["query_type"] == "section"]
    if query_id:
        section_queries = section_queries[section_queries["query_id"] == query_id]

    results: dict[str, list[str]] = {}

    print("\n" + "=" * 78)
    print("  SECTION CANDIDATE INSPECTION (Strictly within Survey References)")
    print("=" * 78)

    for _, qrow in section_queries.iterrows():
        qid = qrow["query_id"]
        qtext = qrow["query_text"]
        sid = qrow["survey_id"]

        survey_meta = papers_df.loc[sid] if sid in papers_df.index else None
        survey_year = float(survey_meta[YEAR]) if (survey_meta is not None and pd.notna(survey_meta[YEAR])) else None
        max_allowed_year = (survey_year + year_tolerance) if survey_year else None

        s_refs = survey_refs.get(sid, [])
        keywords = TOPIC_KEYWORDS.get(qid, [w.lower() for w in re.findall(r"\b\w{4,}\b", qtext.lower())])

        matches = []
        for pid in s_refs:
            if pid not in papers_df.index:
                continue
            p = papers_df.loc[pid]
            pyear = float(p[YEAR]) if pd.notna(p[YEAR]) else None

            # Enforce year constraint
            if max_allowed_year is not None and pyear is not None and pyear > max_allowed_year:
                continue

            text = f"{str(p[TITLE])} {str(p[ABSTRACT])}".lower()
            matched_terms = [kw for kw in keywords if kw in text]
            score = len(matched_terms)

            # Bonus score if term is in the title
            title_text = str(p[TITLE]).lower()
            title_matches = [kw for kw in keywords if kw in title_text]
            score += len(title_matches) * 2

            if score > 0:
                matches.append({
                    "paper_id": pid,
                    "year": int(pyear) if pyear is not None else "N/A",
                    "title": str(p[TITLE]),
                    "score": score,
                    "terms": list(set(matched_terms)),
                })

        matches.sort(key=lambda x: (x["score"], x["year"] if isinstance(x["year"], int) else 0), reverse=True)
        results[qid] = [m["paper_id"] for m in matches]

        status_flag = "OK" if len(matches) >= min_relevant else f"SHORT (< {min_relevant})"
        print(f"\n[{qid}] ({status_flag}) - {qtext}")
        print(f"  Survey: {sid[:8]} (year: {int(survey_year) if survey_year else 'N/A'}, total refs: {len(s_refs)})")
        print(f"  Matches found in survey refs: {len(matches)}")
        print(f"  Keywords used: {', '.join(keywords[:6])}")
        print("  Top matching papers:")
        if not matches:
            print("    (No matching papers found in survey references)")
        for m in matches[:6]:
            terms_str = ", ".join(m["terms"][:3])
            print(f"    - [{m['paper_id'][:8]}] ({m['year']}) score={m['score']} [{terms_str}]")
            print(f"      {m['title'][:70]}")

    print("\n" + "=" * 78)
    return results


def apply_section_references(
    candidates: dict[str, list[str]],
    queries_csv: Path = QUERIES_CSV,
    min_relevant: int = MIN_RELEVANT,
) -> None:
    """Write inspected candidates into queries.csv for queries meeting min_relevant threshold."""
    queries_df = pd.read_csv(queries_csv, dtype=str).fillna("")
    updated_count = 0

    for idx, row in queries_df.iterrows():
        qid = row["query_id"]
        if row["query_type"] == "section" and qid in candidates:
            cands = candidates[qid]
            if len(cands) >= min_relevant:
                queries_df.at[idx, "relevant_ids"] = ";".join(cands)
                updated_count += 1
                print(f"  Applied {len(cands)} relevant IDs to [{qid}]")
            else:
                print(f"  Skipped [{qid}]: has {len(cands)} candidates (< min_relevant={min_relevant})")

    queries_df.to_csv(queries_csv, index=False)
    print(f"\nUpdated {updated_count} section queries in {queries_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and manage section reference assignments.")
    parser.add_argument("--inspect", action="store_true", help="Inspect candidate papers for section queries.")
    parser.add_argument("--query", type=str, default=None, help="Inspect a specific query_id.")
    parser.add_argument("--apply", action="store_true", help="Apply inspected candidates to queries.csv.")
    args = parser.parse_args()

    candidates = inspect_section_candidates(query_id=args.query)
    if args.apply:
        apply_section_references(candidates)


if __name__ == "__main__":
    main()
