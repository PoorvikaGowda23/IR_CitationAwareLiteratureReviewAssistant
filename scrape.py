"""
Fetches paper metadata and citation edges from the Semantic Scholar Graph API.

Step 2 & 3:
- Seed papers, surveys, and landmarks search.
- 1-hop reference & citation expansion, context sentence extraction,
  and citation graph construction.
"""

import argparse
import json
import logging
from pathlib import Path
import time

import pandas as pd
import requests

import config
import schema

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── API constants ──────────────────────────────────────────────────────────────
BASE_URL = "https://api.semanticscholar.org/graph/v1"  # root of the S2 Graph API
SLEEP_SECS = 1.5    # polite pause between calls (S2 limit is 1 req/s across all endpoints)
MAX_RETRIES = 7     # maximum retry attempts on 429 / 5xx
_BACKOFF_BASE_NO_KEY = 15  # seconds base for unauthenticated retries

# Fields requested for paper search and title match
PAPER_FIELDS = "paperId,title,abstract,year,tldr,externalIds,citationCount"

# Default seed queries for GNN-based recommendation retrieval
DEFAULT_QUERIES: list[str] = [
    "graph neural network recommendation",
    "graph collaborative filtering",
    "knowledge graph recommendation",
    "session-based recommendation graph neural network",
    "LightGCN NGCF",
]


# ── 1. Text cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str | None) -> str:
    """Strip leading/trailing whitespace and collapse internal newlines/spaces.

    Args:
        text: Input text string or None.

    Returns:
        Cleaned string with single spaces between words, or empty string if None.
    """
    if not text:
        return ""
    # split() splits on any whitespace (including \n, \r, \t) and collapses them
    return " ".join(text.split())


# ── 2. Request helper ──────────────────────────────────────────────────────────

def s2_get(url: str, params: dict) -> dict:
    """Send a GET request to the Semantic Scholar API with retry logic.

    Attaches the x-api-key header if config.S2_API_KEY is set.
    Retries with back-off on HTTP 429 (rate-limit) and 5xx errors.
    Sleeps SLEEP_SECS after every successful call to remain under rate limits.

    Args:
        url:    Full endpoint URL.
        params: Query parameters dictionary.

    Returns:
        Parsed JSON response body as a Python dict.

    Raises:
        requests.HTTPError: if a non-retryable HTTP error occurs.
        RuntimeError:       if all retry attempts are exhausted.
    """
    headers = {}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY

    backoff_base = 2 if config.S2_API_KEY else _BACKOFF_BASE_NO_KEY

    for attempt in range(MAX_RETRIES):
        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            time.sleep(SLEEP_SECS)  # polite pause after successful request
            return response.json()

        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                wait = int(retry_after)
            else:
                wait = backoff_base * (2 ** attempt)  # exponential backoff
            log.warning(
                "HTTP %d on attempt %d/%d — retrying in %ds",
                response.status_code, attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue

        response.raise_for_status()

    raise RuntimeError(f"All {MAX_RETRIES} retries exhausted for URL: {url}")


# ── 3. Exact title match helper ────────────────────────────────────────────────

def fetch_by_title(title: str) -> dict | None:
    """Search for a paper by exact title match using /paper/search/match.

    Prints asked vs. matched title for visual inspection.

    Args:
        title: Target paper title string.

    Returns:
        Best-matched paper dictionary returned by S2, or None if no match.
    """
    params = {
        "query": title,
        "fields": PAPER_FIELDS,
    }
    data = s2_get(f"{BASE_URL}/paper/search/match", params)
    papers = data.get("data", [])

    if papers:
        best_match = papers[0]
        returned_title = clean_text(best_match.get("title"))
        print(f"asked: {title} | matched: {returned_title}")
        return best_match

    log.warning("No match found for asked title: %r", title)
    return None


# ── 4. Survey fetcher ──────────────────────────────────────────────────────────

def fetch_surveys(
    titles: list[str] | None = None,
    unfound: list[str] | None = None,
) -> list[dict]:
    """Fetch survey papers from SURVEY_TITLES and set is_survey = True.

    Surveys are kept even if their abstract is missing.

    Args:
        titles:  Optional list of survey titles; defaults to config.SURVEY_TITLES.
        unfound: Optional list to append titles that could not be matched.

    Returns:
        List of paper row dictionaries with is_survey = True.
    """
    if titles is None:
        titles = config.SURVEY_TITLES

    surveys: list[dict] = []
    for title in titles:
        paper = fetch_by_title(title)
        if paper is None:
            if unfound is not None:
                unfound.append(title)
            continue

        abstract = clean_text(paper.get("abstract"))
        if not abstract:
            log.warning("Survey paper has no abstract: %r", paper.get("title"))

        tldr_text = None
        if paper.get("tldr"):
            tldr_text = clean_text(paper["tldr"].get("text"))

        surveys.append({
            schema.PAPER_ID: paper.get("paperId"),
            schema.TITLE: clean_text(paper.get("title")),
            schema.ABSTRACT: abstract,
            schema.YEAR: paper.get("year"),
            schema.TLDR: tldr_text,
            schema.IS_SURVEY: True,
        })
    return surveys


# ── 5. Landmark papers fetcher ─────────────────────────────────────────────────

def fetch_landmarks(
    titles: list[str] | None = None,
    unfound: list[str] | None = None,
) -> list[dict]:
    """Fetch landmark papers from SEED_TITLES regardless of citation count.

    Args:
        titles:  Optional list of landmark titles; defaults to config.SEED_TITLES.
        unfound: Optional list to append titles that could not be matched.

    Returns:
        List of paper row dictionaries with is_survey = False.
    """
    if titles is None:
        titles = config.SEED_TITLES

    landmarks: list[dict] = []
    for title in titles:
        paper = fetch_by_title(title)
        if paper is None:
            if unfound is not None:
                unfound.append(title)
            continue

        abstract = clean_text(paper.get("abstract"))
        if not abstract:
            log.warning("Landmark paper has no abstract: %r", paper.get("title"))

        tldr_text = None
        if paper.get("tldr"):
            tldr_text = clean_text(paper["tldr"].get("text"))

        landmarks.append({
            schema.PAPER_ID: paper.get("paperId"),
            schema.TITLE: clean_text(paper.get("title")),
            schema.ABSTRACT: abstract,
            schema.YEAR: paper.get("year"),
            schema.TLDR: tldr_text,
            schema.IS_SURVEY: False,
        })
    return landmarks


# ── 6. Generic seed search with citation threshold ────────────────────────────

def search_seed_papers(
    queries: list[str],
    per_query: int = 20,
    min_citations: int = config.MIN_CITATIONS,
) -> list[dict]:
    """Search Semantic Scholar for seed papers matching each query string.

    Filters papers to only keep those with citationCount >= min_citations
    and a non-empty abstract.

    Args:
        queries:       List of natural-language search query strings.
        per_query:     Maximum number of results to fetch per query.
        min_citations: Minimum citation count threshold.

    Returns:
        List of paper row dictionaries with is_survey = False.
    """
    rows: list[dict] = []

    for query in queries:
        log.info("Searching: %r (limit=%d, min_citations=%d)", query, per_query, min_citations)
        params = {
            "query": query,
            "limit": per_query,
            "fields": PAPER_FIELDS,
        }

        data = s2_get(f"{BASE_URL}/paper/search", params)
        papers = data.get("data", [])
        log.info("  → %d raw results", len(papers))

        for p in papers:
            citations = p.get("citationCount") or 0
            if citations < min_citations:
                continue  # discard papers below citation threshold

            abstract = clean_text(p.get("abstract"))
            if not abstract:
                continue  # discard papers without abstract

            tldr_text = None
            if p.get("tldr"):
                tldr_text = clean_text(p["tldr"].get("text"))

            rows.append({
                schema.PAPER_ID: p.get("paperId"),
                schema.TITLE: clean_text(p.get("title")),
                schema.ABSTRACT: abstract,
                schema.YEAR: p.get("year"),
                schema.TLDR: tldr_text,
                schema.IS_SURVEY: False,
            })

    return rows


# ── 7. Pagination helper for citations & references ───────────────────────────

def fetch_citations_or_references(
    paper_id: str,
    relation: str,
    limit: int = 100,
) -> list[dict]:
    """Fetch all references or citations for a given paper handling offset pagination.

    Args:
        paper_id: Semantic Scholar paper identifier string.
        relation: "references" or "citations".
        limit:    Page size limit per API call (maximum 100).

    Returns:
        List of citation/reference dictionaries returned by S2.
    """
    # For references, the connected paper is citedPaper; for citations, it is citingPaper
    sub_key = "citedPaper" if relation == "references" else "citingPaper"
    fields = f"contexts,intents,isInfluential,{sub_key}.paperId,{sub_key}.title,{sub_key}.abstract,{sub_key}.year"

    all_items: list[dict] = []
    offset = 0

    while True:
        params = {
            "fields": fields,
            "offset": offset,
            "limit": limit,
        }
        try:
            data = s2_get(f"{BASE_URL}/paper/{paper_id}/{relation}", params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.warning("Paper %s not found on S2 for %s", paper_id, relation)
                break
            raise

        items = data.get("data", [])
        if not items:
            break

        all_items.extend(items)

        # S2 provides 'next' offset when additional pages exist
        if "next" in data:
            offset = data["next"]
        else:
            offset += len(items)
            if len(items) < limit:
                break

    return all_items


# ── 8. Edge row extraction with context unrolling ─────────────────────────────

def extract_edge_rows(
    citing_id: str,
    cited_id: str,
    raw_item: dict,
    year_map: dict[str, int | None],
) -> list[dict]:
    """Build edge records unrolled per context sentence.

    Args:
        citing_id: paper_id of the citing paper.
        cited_id:  paper_id of the cited paper.
        raw_item:  Raw S2 edge item with contexts, intents, and isInfluential.
        year_map:  Mapping of paper_id to publication year.

    Returns:
        List of edge dictionaries conforming to schema.EDGES_COLS.
    """
    raw_contexts = raw_item.get("contexts") or []
    # Drop empty or whitespace-only context sentences
    cleaned_contexts = [clean_text(ctx) for ctx in raw_contexts if ctx and clean_text(ctx)]
    if not cleaned_contexts:
        return []

    intents = raw_item.get("intents") or []
    s2_intent = intents[0] if intents else None  # pick primary S2 intent if available
    is_influential = bool(raw_item.get("isInfluential", False))

    citing_year = year_map.get(citing_id)
    cited_year = year_map.get(cited_id)
    if citing_year is not None and cited_year is not None:
        year_gap = int(citing_year) - int(cited_year)
    else:
        year_gap = 0  # default when publication year is unavailable

    rows: list[dict] = []
    for ctx in cleaned_contexts:
        rows.append({
            schema.CITING_ID: citing_id,
            schema.CITED_ID: cited_id,
            schema.CONTEXT_SENTENCE: ctx,
            schema.S2_INTENT: s2_intent,
            schema.IS_INFLUENTIAL: is_influential,
            schema.YEAR_GAP: year_gap,
        })
    return rows


# ── 9. Save survey reference mapping ──────────────────────────────────────────

def save_survey_refs(survey_refs: dict[str, list[str]], path: Path | str) -> None:
    """Save survey references mapping (survey_id -> list of cited paper IDs) to JSON.

    Args:
        survey_refs: Dictionary mapping survey paper IDs to lists of cited paper IDs.
        path:        Destination JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(survey_refs, f, indent=2)


# ── 10. 1-hop Graph expansion ──────────────────────────────────────────────────

def expand_corpus(
    seed_df: pd.DataFrame,
    max_seeds: int | None = None,
    corpus_cap: int = config.CORPUS_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Perform 1-hop citation expansion, survey ref extraction, and edge filtering.

    Args:
        seed_df:    DataFrame of initial seed papers from seed_papers.parquet.
        max_seeds:  Optional maximum number of seed papers to expand (for smoke mode).
        corpus_cap: Upper bound on total unique papers in the corpus.

    Returns:
        Tuple of (papers_df, edges_df, survey_refs).
    """
    # Track papers by paper_id to ensure deduplication
    papers_dict: dict[str, dict] = {}
    for row in seed_df.to_dict(orient="records"):
        pid = row[schema.PAPER_ID]
        papers_dict[pid] = {
            schema.PAPER_ID: pid,
            schema.TITLE: clean_text(row.get(schema.TITLE)),
            schema.ABSTRACT: clean_text(row.get(schema.ABSTRACT)),
            schema.YEAR: row.get(schema.YEAR),
            schema.TLDR: clean_text(row.get(schema.TLDR)),
            schema.IS_SURVEY: bool(row.get(schema.IS_SURVEY, False)),
        }

    year_map: dict[str, int | None] = {
        pid: p.get(schema.YEAR) for pid, p in papers_dict.items()
    }

    # Identify all survey paper IDs in the initial seeds
    survey_ids: set[str] = {
        pid for pid, p in papers_dict.items() if p[schema.IS_SURVEY]
    }

    survey_refs: dict[str, list[str]] = {}
    total_citations_inspected = 0
    citations_with_context = 0
    raw_edges: list[dict] = []

    # 4. Save each survey's full reference list and ensure referenced papers are in corpus
    print("\n--- Collecting Survey Reference Lists ---")
    for s_id in survey_ids:
        s_title = papers_dict[s_id][schema.TITLE]
        log.info("Fetching references for survey: %r", s_title)
        ref_items = fetch_citations_or_references(s_id, "references")
        cited_ids: list[str] = []

        for item in ref_items:
            cited_paper = item.get("citedPaper")
            if not cited_paper or not cited_paper.get("paperId"):
                continue

            c_id = cited_paper["paperId"]
            cited_ids.append(c_id)

            # Ensure survey referenced papers are added to papers even if unlinked elsewhere
            if c_id not in papers_dict:
                papers_dict[c_id] = {
                    schema.PAPER_ID: c_id,
                    schema.TITLE: clean_text(cited_paper.get("title")),
                    schema.ABSTRACT: clean_text(cited_paper.get("abstract")),
                    schema.YEAR: cited_paper.get("year"),
                    schema.TLDR: None,
                    schema.IS_SURVEY: False,
                }
                year_map[c_id] = cited_paper.get("year")

        survey_refs[s_id] = cited_ids
        log.info("  → Recorded %d references for survey %s", len(cited_ids), s_id[:8])

    # Select seeds to expand (capped if max_seeds is provided)
    all_seed_ids = list(seed_df[schema.PAPER_ID])
    seeds_to_expand = all_seed_ids[:max_seeds] if max_seeds else all_seed_ids

    print(f"\n--- Expanding 1-Hop Citations for {len(seeds_to_expand)} Seeds ---")
    for idx, seed_id in enumerate(seeds_to_expand, start=1):
        seed_title = papers_dict[seed_id][schema.TITLE]
        log.info("[%d/%d] Expanding seed: %r (%s)", idx, len(seeds_to_expand), seed_title, seed_id[:8])

        # 1. References (seed cites other papers: seed_id -> cited_id)
        ref_items = fetch_citations_or_references(seed_id, "references")
        for item in ref_items:
            total_citations_inspected += 1
            if item.get("contexts"):
                citations_with_context += 1

            cited_paper = item.get("citedPaper")
            if not cited_paper or not cited_paper.get("paperId"):
                continue

            c_id = cited_paper["paperId"]
            if len(papers_dict) < corpus_cap and c_id not in papers_dict:
                papers_dict[c_id] = {
                    schema.PAPER_ID: c_id,
                    schema.TITLE: clean_text(cited_paper.get("title")),
                    schema.ABSTRACT: clean_text(cited_paper.get("abstract")),
                    schema.YEAR: cited_paper.get("year"),
                    schema.TLDR: None,
                    schema.IS_SURVEY: False,
                }
                year_map[c_id] = cited_paper.get("year")

            raw_edges.extend(extract_edge_rows(seed_id, c_id, item, year_map))

        # 2. Citations (other papers cite seed: citing_id -> seed_id)
        cit_items = fetch_citations_or_references(seed_id, "citations")
        for item in cit_items:
            total_citations_inspected += 1
            if item.get("contexts"):
                citations_with_context += 1

            citing_paper = item.get("citingPaper")
            if not citing_paper or not citing_paper.get("paperId"):
                continue

            c_id = citing_paper["paperId"]
            if len(papers_dict) < corpus_cap and c_id not in papers_dict:
                papers_dict[c_id] = {
                    schema.PAPER_ID: c_id,
                    schema.TITLE: clean_text(citing_paper.get("title")),
                    schema.ABSTRACT: clean_text(citing_paper.get("abstract")),
                    schema.YEAR: citing_paper.get("year"),
                    schema.TLDR: None,
                    schema.IS_SURVEY: False,
                }
                year_map[c_id] = citing_paper.get("year")

            raw_edges.extend(extract_edge_rows(c_id, seed_id, item, year_map))

    # Convert to DataFrame
    papers_df = pd.DataFrame(list(papers_dict.values()))
    papers_df = papers_df.drop_duplicates(subset=[schema.PAPER_ID], keep="first").reset_index(drop=True)

    edges_df = pd.DataFrame(raw_edges)
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=schema.EDGES_COLS)
    else:
        # 3. Drop edges with empty context and edges where either paper is outside the corpus
        corpus_ids = set(papers_df[schema.PAPER_ID])
        edges_df = edges_df[edges_df[schema.CONTEXT_SENTENCE].str.strip() != ""]
        edges_df = edges_df[
            edges_df[schema.CITING_ID].isin(corpus_ids) & edges_df[schema.CITED_ID].isin(corpus_ids)
        ]

        # 5. Remove survey papers and all their outgoing (and incoming) edges from edges.parquet
        edges_df = edges_df[~edges_df[schema.CITING_ID].isin(survey_ids)]
        edges_df = edges_df[~edges_df[schema.CITED_ID].isin(survey_ids)]
        edges_df = edges_df.drop_duplicates().reset_index(drop=True)

    # Calculate percentage of citations that had context sentences
    context_pct = (citations_with_context / total_citations_inspected * 100) if total_citations_inspected else 0.0

    print("\n================ EXPANSION SUMMARY ================")
    print(f"Total papers in corpus:     {len(papers_df)} (surveys: {int(papers_df[schema.IS_SURVEY].sum())})")
    print(f"Total valid edges:          {len(edges_df)}")
    print(f"Citations with context:     {citations_with_context}/{total_citations_inspected} ({context_pct:.1f}%)")
    print("\nEdges per S2 intent label:")
    if not edges_df.empty:
        print(edges_df[schema.S2_INTENT].value_counts(dropna=False).to_string())
    else:
        print("  (no edges found)")
    print("===================================================\n")

    return papers_df, edges_df, survey_refs


# ── Main block ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: scrape seed papers, then expand citations to build graph."""
    parser = argparse.ArgumentParser(description="Scrape papers and build citation graph.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: 5 search papers per query, expand 5 seeds.",
    )
    parser.add_argument(
        "--seeds-only",
        action="store_true",
        help="Only search seed papers and save seed_papers.parquet without 1-hop expansion.",
    )
    args = parser.parse_args()

    seed_file = config.DATA_DIR / "seed_papers.parquet"

    # Step 2: Seed search if requested or if seed_papers.parquet does not exist
    if args.seeds_only or not seed_file.exists():
        per_query = 5 if args.smoke else 20
        unfound_titles: list[str] = []

        print("\n--- Fetching Survey Papers ---")
        surveys = fetch_surveys(unfound=unfound_titles)

        print("\n--- Fetching Landmark Papers ---")
        landmarks = fetch_landmarks(unfound=unfound_titles)

        print("\n--- Fetching Seed Query Papers ---")
        search_papers = search_seed_papers(
            DEFAULT_QUERIES,
            per_query=per_query,
            min_citations=config.MIN_CITATIONS,
        )

        all_rows = surveys + landmarks + search_papers
        df_seeds = pd.DataFrame(all_rows)
        df_seeds = df_seeds.dropna(subset=[schema.PAPER_ID])
        df_seeds = df_seeds.sort_values(by=schema.IS_SURVEY, ascending=False)
        df_seeds = df_seeds.drop_duplicates(subset=[schema.PAPER_ID], keep="first").reset_index(drop=True)

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        schema.save_papers(df_seeds, seed_file)
        log.info("Saved %d seed papers → %s", len(df_seeds), seed_file)

        if args.seeds_only:
            return

    # Step 3: Load seed papers and perform 1-hop expansion
    log.info("Loading seed papers from %s", seed_file)
    seed_df = schema.load_papers(seed_file)

    max_seeds = 5 if args.smoke else None
    papers_df, edges_df, survey_refs = expand_corpus(
        seed_df=seed_df,
        max_seeds=max_seeds,
        corpus_cap=config.CORPUS_CAP,
    )

    # 4. Save survey references
    survey_refs_path = config.DATA_DIR / "survey_refs.json"
    save_survey_refs(survey_refs, survey_refs_path)
    log.info("Saved survey reference lists → %s", survey_refs_path)

    # Save expanded papers and edges using schema.py
    papers_path = config.DATA_DIR / "papers.parquet"
    edges_path = config.DATA_DIR / "edges.parquet"

    schema.save_papers(papers_df, papers_path)
    log.info("Saved %d papers → %s", len(papers_df), papers_path)

    schema.save_edges(edges_df, edges_path)
    log.info("Saved %d edges → %s", len(edges_df), edges_path)


if __name__ == "__main__":
    main()
