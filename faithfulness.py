"""
faithfulness.py
---------------
Step 15: Faithfulness evaluation tooling for citation-aware literature reviews.

Functions:
  1. sample_claims(system_outputs, n=40, seed=SEED)
       Sample (sentence, paper_ids) claims from saved system outputs on the same queries.
  2. export_for_judging(sampled_claims, path)
       Write a blinded CSV: claim, cited paper title+abstract, empty 'label' column.
       Rows are shuffled so the judge cannot infer which system wrote each claim.
  3. judge_prompt()
       Return the rubric string for an LLM judge (supported / partially_supported /
       unsupported), with one example of each, requesting JSON output.
  4. llm_judge(sampled_claims, *, confirm_first=True)
       Run the LLM judge (with disk caching) on all sampled claims and compute:
         faithfulness_rate  = supported / total
         lenient_rate       = (supported + partial) / total
       per system. Does NOT run until `confirm_first=False` so the user can
       inspect the claim count first.
  5. agreement(manual_labels, llm_labels)
       Cohen's kappa between manual labels and the LLM judge on the same sample.

Usage (inspect then judge):
    python faithfulness.py --list-claims           # print claim count per system
    python faithfulness.py --export                # write judging CSV
    python faithfulness.py --judge                 # run LLM judge (asks for confirmation)
    python faithfulness.py --judge --yes           # skip confirmation
    python faithfulness.py --agreement path.csv    # compute kappa from annotated CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import config  # loads .env

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = config.RESULTS_DIR
FAITHFULNESS_DIR: Path = RESULTS_DIR / "faithfulness"
JUDGE_CACHE_DIR: Path = FAITHFULNESS_DIR / "judge_cache"
JUDGING_CSV: Path = FAITHFULNESS_DIR / "judging_sheet.csv"
JUDGE_RESULTS_JSON: Path = FAITHFULNESS_DIR / "judge_results.json"

# ── Shared LLM constants (must match generate_baseline for fair comparison) ────
from generate_baseline import (
    _with_retry,
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY_ENV,
    _call_groq,
    _cache_key,
    _load_cache,
    _save_cache,
    _load_papers_lookup,
)

_PAPERS_PATH: Path = config.DATA_DIR / "papers.parquet"

# Label vocabulary
LABEL_SUPPORTED  = "supported"
LABEL_PARTIAL    = "partially_supported"
LABEL_UNSUPPORTED = "unsupported"
VALID_LABELS = {LABEL_SUPPORTED, LABEL_PARTIAL, LABEL_UNSUPPORTED}
LABEL_ERROR = "error"   # judge output unparseable; excluded from rates


# ── 1. sample_claims ──────────────────────────────────────────────────────────

def sample_claims(
    system_outputs: dict[str, list[dict[str, Any]]],
    n: int = 40,
    seed: int = config.SEED,
) -> list[dict[str, Any]]:
    """Sample n claims from saved system outputs, stratified equally across systems.

    Args:
        system_outputs: Mapping of system_name -> list of generate_review() outputs.
            Each output must contain 'claims' (list of {'sentence', 'paper_ids'})
            and 'query'.
        n:    Total number of claims to sample (split equally across systems).
              If a system has fewer than n//n_systems claimable sentences,
              all its claimable sentences are taken.
        seed: Random seed for reproducibility.

    Returns:
        List of claim dicts, each with keys:
          'claim_id'    — unique integer id (stable for a fixed seed).
          'system'      — system name (e.g. 'baseline', 'citation_aware').
          'query'       — original query the review was generated for.
          'sentence'    — the claim sentence.
          'paper_ids'   — list of cited paper IDs in this sentence.
          'primary_pid' — first paper_id (used for faithfulness checking).
    """
    rng = random.Random(seed)
    systems = sorted(system_outputs.keys())
    n_systems = len(systems)
    per_system = max(1, n // n_systems) if n_systems > 0 else n

    pool: dict[str, list[dict[str, Any]]] = {}
    for sys_name in systems:
        candidates: list[dict[str, Any]] = []
        for output in system_outputs[sys_name]:
            query = output.get("query", "")
            for claim in output.get("claims", []):
                pids = claim.get("paper_ids", [])
                sent = claim.get("sentence", "").strip()
                # Only keep claims that actually cite at least one paper
                if pids and sent:
                    candidates.append({
                        "system":      sys_name,
                        "query":       query,
                        "sentence":    sent,
                        "paper_ids":   pids,
                        "primary_pid": pids[0],
                    })
        rng.shuffle(candidates)
        pool[sys_name] = candidates[:per_system]

    # Flatten and assign stable claim_ids
    sampled: list[dict[str, Any]] = []
    claim_id = 0
    for sys_name in systems:
        for item in pool[sys_name]:
            sampled.append({"claim_id": claim_id, **item})
            claim_id += 1

    # Final shuffle so rows don't cluster by system
    rng.shuffle(sampled)
    return sampled


# ── 2. export_for_judging ────────────────────────────────────────────────────

def export_for_judging(
    sampled_claims: list[dict[str, Any]],
    path: Path | str = JUDGING_CSV,
    papers_path: Path | str = _PAPERS_PATH,
    seed: int = config.SEED,
) -> Path:
    """Write a blinded CSV for human or LLM judging.

    The CSV contains one row per (claim, cited paper) pair.
    The 'system' column is intentionally EXCLUDED so judges cannot infer source.
    An empty 'label' column is left for annotation.

    Columns:
      claim_id, query, sentence, cited_paper_id, cited_title, cited_abstract, label

    Args:
        sampled_claims: Output of sample_claims().
        path:           Destination CSV path.
        papers_path:    papers.parquet for title/abstract lookup.
        seed:           Shuffle seed (same as sample_claims for reproducibility).

    Returns:
        Path to the written CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    papers_lookup = _load_papers_lookup(Path(papers_path))

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for item in sampled_claims:
        # Expand one row per cited paper (primary_pid only to keep things manageable)
        pid = item["primary_pid"]
        meta = papers_lookup.get(pid, {})
        title    = str(meta.get("title", "(title unavailable)")).strip()
        abstract = str(meta.get("abstract", "")).strip()
        if len(abstract) > 800:
            abstract = abstract[:797] + "..."

        rows.append({
            "claim_id":       item["claim_id"],
            "query":          item["query"],
            "sentence":       item["sentence"],
            "cited_paper_id": pid,
            "cited_title":    title,
            "cited_abstract": abstract,
            "label":          "",   # to be filled by judge
        })

    rng.shuffle(rows)

    fieldnames = [
        "claim_id", "query", "sentence",
        "cited_paper_id", "cited_title", "cited_abstract", "label",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[faithfulness] Judging sheet written -> {path}  ({len(rows)} rows)")
    return path


# ── 3. judge_prompt ───────────────────────────────────────────────────────────

def judge_prompt() -> str:
    """Return the rubric string for the LLM faithfulness judge.

    The judge receives one claim at a time and must return a JSON object with
    a single key 'label' whose value is one of:
      'supported', 'partially_supported', 'unsupported'

    Returns:
        The full rubric prompt string (system-turn text).
    """
    return (
        "You are an expert scientific fact-checker evaluating whether a claim in a "
        "literature review is supported by its cited paper.\n\n"
        "TASK: Given a CLAIM sentence and the ABSTRACT of the cited paper, determine "
        "whether the claim is supported.\n\n"
        "LABELS:\n"
        "  supported           — The abstract clearly and directly supports the claim.\n"
        "  partially_supported — The abstract is related and partially supports the claim,\n"
        "                        but some details in the claim go beyond what the abstract states.\n"
        "  unsupported         — The abstract does not support the claim, or the claim\n"
        "                        contradicts / misrepresents the abstract.\n\n"
        "EXAMPLES:\n\n"
        "Example 1 (supported):\n"
        "  Claim: \"LightGCN simplifies graph collaborative filtering by removing "
        "feature transformation and non-linear activation.\"\n"
        "  Abstract excerpt: \"We propose LightGCN that includes only the most essential "
        "component in GCN—neighborhood aggregation—for collaborative filtering. "
        "Specifically, we abandon the feature transformation and nonlinear activation.\"\n"
        "  Output: {\"label\": \"supported\"}\n\n"
        "Example 2 (partially_supported):\n"
        "  Claim: \"NGCF achieves state-of-the-art performance on all benchmark datasets "
        "by propagating embeddings across 5 graph layers.\"\n"
        "  Abstract excerpt: \"We propose NGCF, which exploits the user-item graph "
        "structure by propagating embeddings on it. Extensive experiments demonstrate "
        "significant improvements over existing baselines.\"\n"
        "  Output: {\"label\": \"partially_supported\"}\n\n"
        "Example 3 (unsupported):\n"
        "  Claim: \"The paper proves that graph-based methods always outperform "
        "matrix factorization in cold-start scenarios.\"\n"
        "  Abstract excerpt: \"We present a reproducibility study showing that simple "
        "baselines often match reported GNN results when hyperparameters are tuned properly.\"\n"
        "  Output: {\"label\": \"unsupported\"}\n\n"
        "INSTRUCTIONS:\n"
        "  - Base your judgment ONLY on the provided abstract, not on external knowledge.\n"
        "  - Return ONLY a JSON object on a single line, e.g.: {\"label\": \"supported\"}\n"
        "  - Do NOT add any explanation or extra text.\n"
    )


# ── 4. llm_judge ─────────────────────────────────────────────────────────────

def _judge_one_claim(
    claim: dict[str, Any],
    papers_lookup: dict[str, dict[str, Any]],
    rubric: str,
    model: str,
    api_key: str,
    cache_dir: Path,
) -> str:
    """Call the LLM judge on a single claim. Returns a VALID_LABELS string."""
    pid  = claim["primary_pid"]
    meta = papers_lookup.get(pid, {})
    abstract = str(meta.get("abstract", "")).strip()
    title    = str(meta.get("title", "")).strip()

    user_msg = (
        f"CLAIM: {claim['sentence']}\n\n"
        f"CITED PAPER TITLE: {title}\n"
        f"CITED PAPER ABSTRACT: {abstract[:600] if abstract else '(no abstract available)'}"
    )

    # Cache key over rubric + user message so parameter changes bust the cache
    key = _cache_key(rubric + "\n\n" + user_msg)
    cached = _load_cache(cache_dir, key)
    if cached is not None:
        try:
            label = json.loads(cached).get("label", LABEL_UNSUPPORTED)
            if label in VALID_LABELS:
                return label
        except Exception:
            pass

    # Build a minimal messages list
    full_prompt = user_msg  # rubric goes into system turn via _call_groq

    # Temporarily override _call_groq's system message by injecting a wrapper
    import os as _os
    import importlib
    import generate_baseline as _gb

    # Patch: _call_groq accepts the rubric via a custom wrapper
    label = None
    for _ in range(2):
        response_text = _with_retry(lambda: _call_groq_with_system(
            system=rubric,
            user=user_msg,
            model=model,
            api_key=api_key,
        ))
        try:
            label = json.loads(response_text.strip()).get("label")
        except Exception:
            # partially_supported / unsupported must be tested before supported (substring)
            label = next((l for l in (LABEL_PARTIAL, LABEL_UNSUPPORTED, LABEL_SUPPORTED)
                          if l in response_text), None)
        if label in VALID_LABELS:
            break
        label = None

    if label is None:
        # Unparseable even after a retry: report as an error, never cache it
        return LABEL_ERROR

    _save_cache(cache_dir, key, json.dumps({"label": label}))
    return label


def _call_groq_with_system(system: str, user: str, model: str, api_key: str) -> str:
    """Like generate_baseline._call_groq but with an explicit system message."""
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
            max_tokens=512,   # reasoning model: leave room for reasoning + JSON
            reasoning_effort="low",
        )
        return str(completion.choices[0].message.content)
    except ImportError:
        import requests  # type: ignore
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0,
            "max_tokens": 512,
            "reasoning_effort": "low",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API HTTP {resp.status_code}: {resp.text[:300]}")
        return str(resp.json()["choices"][0]["message"]["content"])


def llm_judge(
    sampled_claims: list[dict[str, Any]],
    *,
    model: str = DEFAULT_GROQ_MODEL,
    cache_dir: Path | str = JUDGE_CACHE_DIR,
    papers_path: Path | str = _PAPERS_PATH,
    output_path: Path | str = JUDGE_RESULTS_JSON,
    confirm_first: bool = True,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Run the LLM faithfulness judge on all sampled claims.

    SAFETY: Does NOT call the API until confirm_first=False.
    Always call with confirm_first=True first to print the claim count,
    then re-call with confirm_first=False after user confirmation.

    Args:
        sampled_claims: Output of sample_claims().
        model:          Groq model to use.
        cache_dir:      Directory for caching judge responses.
        papers_path:    papers.parquet path.
        output_path:    JSON file to write labelled results to.
        confirm_first:  If True, print stats and return without calling the API.
        verbose:        Print per-claim progress.

    Returns:
        dict mapping system_name -> {
            'faithfulness_rate': float,   # supported / total
            'lenient_rate':      float,   # (supported + partial) / total
            'n_supported':       int,
            'n_partial':         int,
            'n_unsupported':     int,
            'n_total':           int,
        }
        Returns empty dict if confirm_first=True (no judging performed).
    """
    # Count claims per system before doing anything
    from collections import Counter
    sys_counts: Counter[str] = Counter(c["system"] for c in sampled_claims)

    print("[faithfulness] Claims to judge:")
    for sname, cnt in sorted(sys_counts.items()):
        print(f"  {sname:30s}  {cnt} claims")
    print(f"  {'TOTAL':30s}  {len(sampled_claims)} claims")

    if confirm_first:
        print(
            "\n[faithfulness] confirm_first=True — NOT calling the LLM judge yet.\n"
            "Re-call with confirm_first=False (or use --judge --yes) to proceed."
        )
        return {}

    api_key = os.environ.get(GROQ_API_KEY_ENV, "").strip()
    if not api_key:
        raise EnvironmentError(f"GROQ_API_KEY not set. Add to .env.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    papers_lookup = _load_papers_lookup(Path(papers_path))
    rubric = judge_prompt()

    # Judge every claim
    labelled: list[dict[str, Any]] = []
    for i, claim in enumerate(sampled_claims):
        if verbose:
            print(f"  Judging [{i+1}/{len(sampled_claims)}] claim_id={claim['claim_id']} "
                  f"system={claim['system']}...", end=" ", flush=True)
        label = _judge_one_claim(
            claim=claim,
            papers_lookup=papers_lookup,
            rubric=rubric,
            model=model,
            api_key=api_key,
            cache_dir=cache_dir,
        )
        if verbose:
            print(label)
        labelled.append({**claim, "llm_label": label})

    # Aggregate per system
    from collections import defaultdict
    sys_labels: dict[str, list[str]] = defaultdict(list)
    n_errors: dict[str, int] = defaultdict(int)
    for item in labelled:
        if item["llm_label"] in VALID_LABELS:
            sys_labels[item["system"]].append(item["llm_label"])
        else:
            n_errors[item["system"]] += 1
    if n_errors:
        print(f"[faithfulness] WARNING: judge errors (excluded from rates): {dict(n_errors)}")

    metrics: dict[str, dict[str, float]] = {}
    for sname, labels in sorted(sys_labels.items()):
        n_total      = len(labels)
        n_supported  = labels.count(LABEL_SUPPORTED)
        n_partial    = labels.count(LABEL_PARTIAL)
        n_unsupported = labels.count(LABEL_UNSUPPORTED)
        faith_rate   = n_supported / n_total if n_total else 0.0
        lenient_rate = (n_supported + n_partial) / n_total if n_total else 0.0
        metrics[sname] = {
            "faithfulness_rate": round(faith_rate, 4),
            "lenient_rate":      round(lenient_rate, 4),
            "n_supported":       n_supported,
            "n_partial":         n_partial,
            "n_unsupported":     n_unsupported,
            "n_total":           n_total,
            "n_error":           n_errors.get(sname, 0),
        }

    # Save full labelled output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "labelled_claims": labelled}, f, indent=2)
    print(f"\n[faithfulness] Results saved -> {output_path}")

    # Add llm_label to the blinded sheet (matched by claim_id) for later kappa
    if JUDGING_CSV.exists():
        import pandas as pd
        sheet = pd.read_csv(JUDGING_CSV)
        by_id = {c["claim_id"]: c["llm_label"] for c in labelled}
        sheet["llm_label"] = sheet["claim_id"].map(by_id).fillna("")
        sheet.to_csv(JUDGING_CSV, index=False)

    # Print summary table
    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Print a formatted faithfulness metrics table."""
    print("\n" + "=" * 65)
    print("FAITHFULNESS METRICS (LLM judge)")
    print("=" * 65)
    header = f"{'System':<28} {'Faith%':>7} {'Lenient%':>9} {'Supp':>5} {'Part':>5} {'Uns':>5} {'N':>5}"
    print(header)
    print("-" * 65)
    for sname, m in sorted(metrics.items()):
        print(
            f"{sname:<28} "
            f"{m['faithfulness_rate']*100:>6.1f}% "
            f"{m['lenient_rate']*100:>8.1f}% "
            f"{m['n_supported']:>5} "
            f"{m['n_partial']:>5} "
            f"{m['n_unsupported']:>5} "
            f"{m['n_total']:>5}"
        )
    print("=" * 65)


# ── 5. agreement (Cohen's kappa) ─────────────────────────────────────────────

def agreement(
    manual_labels: list[str],
    llm_labels: list[str],
) -> float:
    """Compute Cohen's kappa between manual labels and LLM judge labels.

    Both lists must have the same length and contain values from VALID_LABELS.

    Args:
        manual_labels: Human-assigned labels (list of strings from VALID_LABELS).
        llm_labels:    LLM-assigned labels  (same length).

    Returns:
        Cohen's kappa (float in [-1, 1]).
    """
    if len(manual_labels) != len(llm_labels):
        raise ValueError(
            f"Label lists must have equal length; got {len(manual_labels)} vs {len(llm_labels)}"
        )
    if not manual_labels:
        raise ValueError("Label lists are empty.")

    classes = sorted(VALID_LABELS)
    n = len(manual_labels)

    # Observed agreement
    p_o = sum(m == l for m, l in zip(manual_labels, llm_labels)) / n

    # Expected agreement
    manual_dist = {c: manual_labels.count(c) / n for c in classes}
    llm_dist    = {c: llm_labels.count(c) / n for c in classes}
    p_e = sum(manual_dist[c] * llm_dist[c] for c in classes)

    if abs(1.0 - p_e) < 1e-12:
        # Perfect expected agreement → kappa is undefined; return 1.0 if p_o == 1.0 else 0.0
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(kappa, 4)


def agreement_from_csv(
    annotated_csv: Path | str,
    manual_col: str = "label",
    llm_col: str = "llm_label",
) -> float:
    """Load manual and LLM labels from a CSV and compute Cohen's kappa.

    The CSV is expected to have columns ``manual_col`` and ``llm_col``.
    Rows where either label is empty or invalid are skipped.

    Args:
        annotated_csv: Path to a CSV that has both manual and LLM labels.
        manual_col:    Column name for manual labels.
        llm_col:       Column name for LLM labels.

    Returns:
        Cohen's kappa.
    """
    annotated_csv = Path(annotated_csv)
    manual: list[str] = []
    llm: list[str] = []
    skipped = 0

    with open(annotated_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get(manual_col, "").strip()
            l = row.get(llm_col, "").strip()
            if m in VALID_LABELS and l in VALID_LABELS:
                manual.append(m)
                llm.append(l)
            else:
                skipped += 1

    if skipped:
        print(f"[faithfulness] Skipped {skipped} rows with missing/invalid labels.")
    if not manual:
        raise ValueError("No valid paired labels found in the CSV.")

    kappa = agreement(manual, llm)
    print(f"[faithfulness] Cohen's kappa = {kappa:.4f}  (n={len(manual)} pairs)")
    return kappa


# ── Helpers: load judge results ───────────────────────────────────────────────

def load_judge_results(
    path: Path | str = JUDGE_RESULTS_JSON,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Load previously saved judge results from disk.

    Returns:
        (metrics dict, labelled_claims list)
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["metrics"], data["labelled_claims"]


# ── CLI ───────────────────────────────────────────────────────────────────────

DEMO_QUERY_IDS = ["s1_gcf", "s1_session", "s2_kg", "s2_contrastive", "s3_dynamic"]


def _run_demo_sample() -> list[dict[str, Any]]:
    """Generate a demo sample from cached review outputs for the default query."""
    from generate_baseline import generate_review as gen_baseline
    from generate import generate_review as gen_intent
    from retrieval import retrieve

    import pandas as pd
    qdf = pd.read_csv(config.DATA_DIR / "queries.csv").set_index("query_id")
    queries = [qdf.loc[q, "query_text"] for q in DEMO_QUERY_IDS]

    baseline_outs, intent_outs = [], []
    for query in queries:
        print(f"[faithfulness] Query: {query!r}")
        ranked = retrieve(query, k=10, surface_contradictions=False)
        baseline_outs.append(gen_baseline(query, ranked, k=10))
        intent_outs.append(gen_intent(query, ranked, k=10))

    system_outputs = {
        "baseline":       baseline_outs,
        "citation_aware": intent_outs,
    }
    sampled = sample_claims(system_outputs, n=40)
    return sampled


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Step 15: Faithfulness evaluation for citation-aware review systems."
    )
    parser.add_argument(
        "--list-claims", action="store_true",
        help="Print the number of claims per system and exit (does not call LLM).",
    )
    parser.add_argument(
        "--export", action="store_true",
        help=f"Export blinded judging CSV to {JUDGING_CSV}.",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run the LLM judge (asks for confirmation unless --yes is also passed).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the claim-count confirmation prompt (use with --judge).",
    )
    parser.add_argument(
        "--agreement", metavar="CSV",
        help="Compute Cohen's kappa from an annotated CSV (needs 'label' and 'llm_label' cols).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_GROQ_MODEL,
        help=f"Groq model for LLM judge (default: {DEFAULT_GROQ_MODEL}).",
    )
    args = parser.parse_args()

    # Agreement from an existing CSV — no need for retrieval/generation
    if args.agreement:
        agreement_from_csv(args.agreement)
        return

    if not any([args.list_claims, args.export, args.judge]):
        parser.print_help()
        return

    sampled = _run_demo_sample()

    if args.list_claims:
        from collections import Counter
        counts = Counter(c["system"] for c in sampled)
        print("\n[faithfulness] Claim counts:")
        for sname, cnt in sorted(counts.items()):
            print(f"  {sname}: {cnt}")
        print(f"  TOTAL: {len(sampled)}")

    if args.export:
        FAITHFULNESS_DIR.mkdir(parents=True, exist_ok=True)
        export_for_judging(sampled)

    if args.judge:
        confirm = not args.yes
        metrics = llm_judge(
            sampled,
            model=args.model,
            confirm_first=confirm,
        )
        if metrics:
            _print_metrics(metrics)


if __name__ == "__main__":
    main()
