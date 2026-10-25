"""
generate_baseline.py
--------------------
Step 14a: Baseline review generator.

Implements generate_review(query, ranked) using a Groq LLM.

Pipeline:
  1. Build a prompt from the top-k papers' title, year, and abstract (flat text,
     no citation intent labels).
  2. Call the Groq API (key from GROQ_API_KEY env var) with temperature=0 and ask
     for a 3-4 paragraph literature review citing papers inline as [S2:paper_id].
  3. Post-process the output:
     - Split into sentences.
     - Extract [S2:...] tags with a regex.
     - Return {'text': ..., 'claims': [{'sentence': ..., 'paper_ids': [...]}]}.
  4. Flag any tag whose id is not in the ranked list; report the count.
  5. Cache responses to disk so repeated identical queries are free.

Usage:
    python generate_baseline.py --query "Graph neural networks for recommendation" -k 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# Import config to trigger .env loading (populates os.environ)
import config  # noqa: F401  (side-effect: loads .env)

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_API_KEY_ENV: str = "GROQ_API_KEY"
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"
CACHE_DIR: Path = config.RESULTS_DIR / "review_cache"
_S2_TAG_RE: re.Pattern[str] = re.compile(r"\[S2:([^\]]+)\]")
_SENT_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")


# ── Prompt building ────────────────────────────────────────────────────────────

def _build_prompt(query: str, papers: list[dict[str, Any]], top_k: int) -> str:
    """Construct the user prompt for the review LLM call.

    Args:
        query:  The original user query string.
        papers: Rows from papers.parquet for the top-k ranked paper IDs,
                each expected to have 'paper_id', 'title', 'year', 'abstract'.
        top_k:  Maximum number of papers included in the prompt.

    Returns:
        A fully formatted prompt string.
    """
    paper_lines: list[str] = []
    for i, p in enumerate(papers[:top_k], start=1):
        pid   = str(p.get("paper_id", ""))
        title = str(p.get("title", "")).strip() or "(no title)"
        year  = p.get("year", "n.d.")
        abst  = str(p.get("abstract", "")).strip()
        if len(abst) > 600:
            abst = abst[:597] + "..."
        paper_lines.append(
            f"[{i}] ID={pid}  ({year})  {title}\n"
            f"    Abstract: {abst}"
        )

    papers_section = "\n\n".join(paper_lines)

    prompt = textwrap.dedent(f"""\
        You are an expert academic writer.
        Write a concise literature review (3-4 paragraphs) on the following topic:

        TOPIC: {query}

        Use ONLY the papers listed below as sources. After every claim you make, cite the
        relevant paper(s) inline using the exact format [S2:<ID>] where <ID> is the paper's
        ID shown above. You may cite multiple papers for one claim: [S2:id1][S2:id2].
        Do NOT invent citations or IDs that are not in the list.

        --- PAPERS ---
        {papers_section}
        --- END OF PAPERS ---

        Write the review now, following all instructions above.
    """)
    return prompt


# ── Disk cache ─────────────────────────────────────────────────────────────────

def _cache_key(prompt: str) -> str:
    """Return a SHA-256 hex digest of the prompt to use as a cache filename."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_cache(cache_dir: Path, key: str) -> str | None:
    """Return cached LLM response text, or None if not cached."""
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return str(data["response"])
        except Exception:
            return None
    return None


def _save_cache(cache_dir: Path, key: str, response: str) -> None:
    """Persist an LLM response to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"response": response}, f, ensure_ascii=False, indent=2)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _with_retry(fn, *, attempts: int = 6):
    """Call fn(); on a Groq 429 / rate-limit error wait and retry."""
    import re
    import time
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if attempt == attempts - 1 or not ("429" in msg or "rate limit" in msg.lower()):
                raise
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 15 * (attempt + 1)
            print(f"[groq] rate limited, sleeping {wait:.0f}s (attempt {attempt + 1}/{attempts})")
            time.sleep(wait)


def _call_groq(prompt: str, model: str, api_key: str) -> str:
    return _with_retry(lambda: _call_groq_once(prompt, model, api_key))


def _call_groq_once(prompt: str, model: str, api_key: str) -> str:
    """Call the Groq Chat Completions API and return the assistant text.

    Uses the groq Python SDK if available; falls back to raw requests.

    Args:
        prompt:  The user-turn message text.
        model:   Groq model identifier.
        api_key: Groq API key.

    Returns:
        The assistant reply text.

    Raises:
        RuntimeError: If the API call fails.
    """
    system_msg = (
        "You are an expert academic writer who writes accurate, "
        "well-structured literature reviews with inline citations."
    )

    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2048,
        )
        return str(completion.choices[0].message.content)

    except ImportError:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Neither 'groq' nor 'requests' is installed. "
                "Run: pip install groq"
            ) from exc

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Groq API returned HTTP {resp.status_code}: {resp.text[:400]}"
            )
        return str(resp.json()["choices"][0]["message"]["content"])


# ── Post-processing ────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_claims(
    text: str,
    ranked_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Parse the review text and return per-sentence claim structures.

    Args:
        text:       Raw review text from the LLM.
        ranked_ids: Set of paper_ids that were actually ranked / passed in.

    Returns:
        (claims, n_invalid) where:
          claims    — list of {'sentence': str, 'paper_ids': list[str]}
          n_invalid — number of [S2:...] tags whose id was not in ranked_ids
    """
    sentences = _split_sentences(text)
    claims: list[dict[str, Any]] = []
    n_invalid = 0

    for sent in sentences:
        tags = _S2_TAG_RE.findall(sent)
        valid_ids: list[str] = []
        for tag_id in tags:
            tag_id = tag_id.strip()
            if tag_id in ranked_ids:
                valid_ids.append(tag_id)
            else:
                n_invalid += 1
        claims.append({"sentence": sent, "paper_ids": valid_ids})

    return claims, n_invalid


# ── Paper metadata loader (lazy, in-memory cache) ──────────────────────────────

_PAPERS_DF_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def _load_papers_lookup(
    papers_path: Path = config.DATA_DIR / "papers.parquet",
) -> dict[str, dict[str, Any]]:
    """Return paper_id -> {title, year, abstract} lookup dict."""
    key = str(papers_path)
    if key not in _PAPERS_DF_CACHE:
        import pandas as pd
        df = pd.read_parquet(papers_path)
        lookup: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            pid = str(row.get("paper_id", ""))
            lookup[pid] = {
                "paper_id": pid,
                "title":    str(row.get("title", "")),
                "year":     row.get("year", None),
                "abstract": str(row.get("abstract", "")),
                "tldr":     str(row.get("tldr", "") or ""),
            }
        _PAPERS_DF_CACHE[key] = lookup
    return _PAPERS_DF_CACHE[key]


# ── Main public interface ──────────────────────────────────────────────────────

def generate_review(
    query: str,
    ranked: list[dict[str, Any]],
    *,
    k: int = 10,
    model: str = DEFAULT_GROQ_MODEL,
    cache_dir: Path | str = CACHE_DIR,
    papers_path: Path | str = config.DATA_DIR / "papers.parquet",
    verbose: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Generate a baseline literature review from a ranked list of papers.

    This is the shared-contract entry point for Step 14a.

    Args:
        query:       Natural-language query driving the review.
        ranked:      List of retrieval-contract dicts (paper_id, score, rank, …).
                     Only the first ``k`` entries are used.
        k:           Number of top-ranked papers to include in the prompt.
        model:       Groq model to use.
        cache_dir:   Directory for caching LLM responses.
        papers_path: Path to papers.parquet for title/year/abstract metadata.
        verbose:     If True, print progress messages to stdout.
        use_cache:   If False, skip the disk cache (no read, no write).

    Returns:
        dict with keys:
          'text'           — Full generated review text.
          'claims'         — list of {'sentence': str, 'paper_ids': list[str]}.
          'n_invalid_tags' — Count of [S2:...] tags not in the ranked list.
          'model'          — Groq model used.
          'query'          — Original query.
          'paper_ids_used' — Ordered list of paper IDs passed to the LLM.
          'from_cache'     — True if the response came from the disk cache.

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

    # Resolve top-k paper IDs
    top_k_entries = ranked[:k]
    ranked_ids_ordered = [str(r["paper_id"]) for r in top_k_entries]
    ranked_ids_set = set(ranked_ids_ordered)

    # Load paper metadata
    if not papers_path.exists():
        raise FileNotFoundError(f"papers.parquet not found at {papers_path}")

    papers_lookup = _load_papers_lookup(papers_path)
    papers_for_prompt: list[dict[str, Any]] = []
    for pid in ranked_ids_ordered:
        if pid in papers_lookup:
            papers_for_prompt.append(papers_lookup[pid])
        else:
            # Graceful fallback: minimal record so the LLM can still cite the paper
            papers_for_prompt.append(
                {"paper_id": pid, "title": "(title unavailable)", "year": "n.d.", "abstract": ""}
            )

    # Build prompt
    prompt = _build_prompt(query, papers_for_prompt, top_k=k)

    # Cache lookup
    cache_key = _cache_key(prompt)
    cached_text = _load_cache(cache_dir, cache_key) if use_cache else None
    from_cache = cached_text is not None

    if from_cache:
        if verbose:
            print(f"[generate_baseline] Cache HIT ({cache_key[:12]}…)")
        review_text = cached_text
    else:
        if verbose:
            print(
                f"[generate_baseline] Cache MISS — calling Groq "
                f"(model={model}, papers={len(papers_for_prompt)})…"
            )
        review_text = _call_groq(prompt, model=model, api_key=api_key)
        if use_cache:
            _save_cache(cache_dir, cache_key, review_text)
            if verbose:
                print(
                    f"[generate_baseline] Cached -> "
                    f"{cache_dir / (cache_key + '.json')}"
                )

    # Post-process
    claims, n_invalid = _extract_claims(review_text, ranked_ids_set)

    if n_invalid > 0:
        print(
            f"[generate_baseline] WARNING: {n_invalid} citation tag(s) reference "
            f"paper IDs not in the ranked list (hallucinated or out-of-range)."
        )

    return {
        "text":            review_text,
        "claims":          claims,
        "n_invalid_tags":  n_invalid,
        "model":           model,
        "query":           query,
        "paper_ids_used":  ranked_ids_ordered,
        "from_cache":      from_cache,
    }


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    # ── Force UTF-8 output so LLM Unicode chars don't crash cp1252 terminals ──
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Step 14a: Baseline literature review generator (Groq LLM)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Graph neural networks for collaborative filtering in recommender systems",
        help="Query string to generate a review for.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of top-k papers to include in the prompt (default 10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GROQ_MODEL,
        help=f"Groq model identifier (default: {DEFAULT_GROQ_MODEL}).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the disk cache and always call the API.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    args = parser.parse_args()

    print(f"Query: {args.query!r}")
    print("Running retrieval (stage1 + graph re-ranking)…")

    from retrieval import retrieve  # imported here to avoid circular import at module level
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

    print("\n" + "=" * 80)
    print("GENERATED REVIEW")
    print("=" * 80)
    print(result["text"])
    print("=" * 80)

    print(f"\nPapers passed to LLM ({len(result['paper_ids_used'])}):")
    for pid in result["paper_ids_used"]:
        print(f"  • {pid}")

    n_sent      = len(result["claims"])
    n_cited     = sum(1 for c in result["claims"] if c["paper_ids"])
    n_valid_tag = sum(len(c["paper_ids"]) for c in result["claims"])
    n_invalid   = result["n_invalid_tags"]

    print(f"\nTotal sentences:     {n_sent}")
    print(f"Sentences with cite: {n_cited}")
    print(f"Total valid tags:    {n_valid_tag}")
    if n_invalid > 0:
        print(f"Invalid tags:        {n_invalid}  ← hallucinated IDs!")
    else:
        print("Invalid tags:        0  ✓ no hallucinated IDs")
    print(f"From cache:          {result['from_cache']}")
    print(f"Model:               {result['model']}")

    print("\n--- PER-SENTENCE CLAIMS ---")
    for i, claim in enumerate(result["claims"], 1):
        if claim["paper_ids"]:
            print(f"[{i}] {claim['sentence']}")
            print(f"     ↳ cites: {claim['paper_ids']}")


if __name__ == "__main__":
    main()
