"""
Fetches paper metadata from the Semantic Scholar Graph API and saves to data/.

Step 2 (first half): seed-paper search only.
Functions: s2_get, search_seed_papers, mark_surveys.
"""

import argparse
import time
import logging

import pandas as pd
import requests

import config
import schema

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── API constants ──────────────────────────────────────────────────────────────
BASE_URL   = "https://api.semanticscholar.org/graph/v1"   # root of the S2 Graph API
SLEEP_SECS = 3.0    # polite pause between every successful API call (seconds)
MAX_RETRIES = 7     # maximum number of retry attempts on 429 / 5xx
# Base wait multiplier for 429 back-off when no API key is set (unauthenticated
# requests are rate-limited much more aggressively by S2)
_BACKOFF_BASE_NO_KEY = 15   # seconds; doubles each retry: 15, 30, 60, 120 …

# Fields we request from /paper/search — must map to schema column names
PAPER_FIELDS = "paperId,title,abstract,year,tldr,externalIds,citationCount"

# Default seed queries for GNN-based recommendation retrieval
DEFAULT_QUERIES: list[str] = [
    "graph neural network recommendation",
    "graph collaborative filtering",
    "knowledge graph recommendation",
    "session-based recommendation graph neural network",
    "LightGCN NGCF",
]

# Titles of known survey papers to flag with is_survey = True
DEFAULT_SURVEY_TITLES: list[str] = [
    "Graph Neural Networks in Recommender Systems: A Survey",
    "A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions",
    "Graph Learning based Recommender Systems: A Review",
]


# ── 1. Request helper ──────────────────────────────────────────────────────────

def s2_get(url: str, params: dict) -> dict:
    """Send a GET request to the Semantic Scholar API with retry logic.

    Attaches the x-api-key header if config.S2_API_KEY is set.
    Retries with exponential back-off on HTTP 429 (rate-limit) and 5xx errors.
    Sleeps SLEEP_SECS after every successful call to be a polite client.

    Args:
        url:    Full endpoint URL (e.g. BASE_URL + "/paper/search").
        params: Query-string parameters dict passed to requests.get.

    Returns:
        Parsed JSON response body as a Python dict.

    Raises:
        requests.HTTPError: if a non-retryable HTTP error occurs.
        RuntimeError:       if all retry attempts are exhausted.
    """
    # Build headers — only include the key header when a key is available
    headers = {}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY

    # Choose the back-off base depending on whether we have a key.
    # Without a key the API rate-limits far more aggressively.
    backoff_base = 2 if config.S2_API_KEY else _BACKOFF_BASE_NO_KEY

    for attempt in range(MAX_RETRIES):
        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            time.sleep(SLEEP_SECS)          # polite pause after a successful call
            return response.json()

        if response.status_code == 429 or response.status_code >= 500:
            # Prefer the server-supplied Retry-After header when present
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                wait = int(retry_after)     # S2 returns an integer number of seconds
            else:
                # Exponential back-off: backoff_base * 2^attempt
                # No-key example: 15, 30, 60, 120, 240, 480, 960 s
                # With-key example: 2,  4,  8,  16,  32,  64, 128 s
                wait = backoff_base * (2 ** attempt)
            log.warning(
                "HTTP %d on attempt %d/%d — retrying in %ds",
                response.status_code, attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue                        # go to the next attempt

        # Any other 4xx is a real error; raise immediately
        response.raise_for_status()

    raise RuntimeError(
        f"All {MAX_RETRIES} retries exhausted for URL: {url}"
    )


# ── 2. Seed-paper search ───────────────────────────────────────────────────────

def search_seed_papers(
    queries: list[str],
    per_query: int = 20,
) -> pd.DataFrame:
    """Search Semantic Scholar for seed papers matching each query string.

    Calls /paper/search for every query, collects results, deduplicates on
    paper_id, and drops any paper that has no abstract text.

    Args:
        queries:   List of natural-language query strings.
        per_query: Maximum number of results to request per query (API limit: 100).

    Returns:
        DataFrame with columns matching schema.PAPERS_COLS, deduplicated,
        with abstracts guaranteed to be non-empty strings.
    """
    rows: list[dict] = []   # accumulate raw result dicts here

    for query in queries:
        log.info("Searching: %r (limit=%d)", query, per_query)

        # Build the API call parameters
        params = {
            "query":  query,
            "limit":  per_query,
            "fields": PAPER_FIELDS,
        }

        data = s2_get(f"{BASE_URL}/paper/search", params)
        papers = data.get("data", [])       # list of paper objects from the API

        log.info("  → %d results", len(papers))

        for p in papers:
            # Flatten the nested tldr object to a plain string (or None)
            tldr_text = None
            if p.get("tldr"):
                tldr_text = p["tldr"].get("text")   # S2 returns {"model","text"}

            rows.append({
                schema.PAPER_ID:  p.get("paperId"),
                schema.TITLE:     p.get("title"),
                schema.ABSTRACT:  p.get("abstract"),
                schema.YEAR:      p.get("year"),
                schema.TLDR:      tldr_text,
                schema.IS_SURVEY: False,    # default; mark_surveys sets True later
            })

    df = pd.DataFrame(rows)

    # Drop papers with a missing paper_id (shouldn't happen, but guard anyway)
    df = df.dropna(subset=[schema.PAPER_ID])

    # Deduplicate: keep the first occurrence of each paper_id
    df = df.drop_duplicates(subset=[schema.PAPER_ID], keep="first")

    # Keep only papers with a non-empty abstract string
    df = df[
        df[schema.ABSTRACT].notna() &          # not NaN
        (df[schema.ABSTRACT].str.strip() != "")  # not blank
    ]

    # Reset index so it runs 0..N after filtering
    df = df.reset_index(drop=True)

    log.info("Seed corpus: %d unique papers with abstracts", len(df))
    return df


# ── 3. Survey flagging ─────────────────────────────────────────────────────────

def mark_surveys(
    papers_df: pd.DataFrame,
    survey_titles: list[str] | None = None,
) -> pd.DataFrame:
    """Set is_survey = True for papers whose title matches any entry in survey_titles.

    Matching is case-insensitive and strips leading/trailing whitespace.
    The DataFrame is modified in-place and also returned for chaining.

    Args:
        papers_df:     DataFrame with at least schema.TITLE and schema.IS_SURVEY columns.
        survey_titles: List of exact (but case-insensitive) survey paper titles.
                       Defaults to DEFAULT_SURVEY_TITLES if None.

    Returns:
        The same DataFrame with is_survey updated.
    """
    if survey_titles is None:
        survey_titles = DEFAULT_SURVEY_TITLES

    # Normalise the target titles once, outside the loop
    normalised_targets = {t.strip().lower() for t in survey_titles}

    # Build a boolean mask: True where the paper title matches a survey title
    mask = papers_df[schema.TITLE].str.strip().str.lower().isin(normalised_targets)

    papers_df.loc[mask, schema.IS_SURVEY] = True   # flag matched rows

    n_flagged = int(mask.sum())
    log.info("Flagged %d survey paper(s) in the corpus", n_flagged)
    return papers_df


# ── Main block ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: search seed papers and save to data/seed_papers.parquet.

    Pass --smoke to run a quick test with only 5 papers per query.
    """
    parser = argparse.ArgumentParser(description="Scrape seed papers from Semantic Scholar.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: fetch only 5 papers per query.",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Tiny mode: fetch only 2 papers per query (for rate-limited / keyless runs).",
    )
    args = parser.parse_args()

    # Priority: --tiny beats --smoke beats normal
    if args.tiny:
        per_query = 2
    elif args.smoke:
        per_query = 5
    else:
        per_query = 20     # full run

    # Step 1: fetch papers
    papers_df = search_seed_papers(DEFAULT_QUERIES, per_query=per_query)

    # Step 2: flag surveys
    papers_df = mark_surveys(papers_df, DEFAULT_SURVEY_TITLES)

    # Step 3: validate schema before writing
    out_path = config.DATA_DIR / "seed_papers.parquet"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)   # create data/ if absent
    schema.save_papers(papers_df, out_path)

    log.info("Saved %d papers → %s", len(papers_df), out_path)


if __name__ == "__main__":
    main()
