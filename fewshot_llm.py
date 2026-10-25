"""
Step 17a: few-shot LLM citation-intent classifier + hand-labelling helpers.

  classify_fewshot()          -- label SciCite sentences with an LLM (cached on disk)
  eval_fewshot()              -- score it on a seeded 300-sentence SciCite test sample
  export_label_candidates()   -- write data/to_label.csv for hand labelling
  load_hand_labels()          -- read the filled CSV -> data/hand_labels.parquet

Usage:
    python fewshot_llm.py --smoke              # 5 sentences only
    python fewshot_llm.py --eval --n 300       # LLM calls (confirm cost first!)
    python fewshot_llm.py --export --n 300     # no LLM calls
    python fewshot_llm.py --load-labels
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import schema
from config import DATA_DIR, RESULTS_DIR, SEED
from generate_baseline import (
    DEFAULT_GROQ_MODEL,
    GROQ_API_KEY_ENV,
    _cache_key,
    _call_groq,
    _load_cache,
    _save_cache,
)
from intent_classifier import LABEL_NAMES

FEWSHOT_CACHE_DIR: Path = RESULTS_DIR / "fewshot_cache"   # one JSON file per LLM response
FEWSHOT_METRICS: Path = RESULTS_DIR / "fewshot_intent.json"
TO_LABEL_CSV: Path = DATA_DIR / "to_label.csv"
HAND_LABELS_PARQUET: Path = DATA_DIR / "hand_labels.parquet"
HAND_LABEL_CHOICES: set[str] = {"extension", "contradiction", "other"}

# Phrases that hint at an extension or contradiction citation
CUE_PHRASES: list[str] = [
    "however", "in contrast", "unlike", "fails to",
    "outperforms", "extends", "builds on", "improves upon",
]
CUE_FRACTION_OF_SAMPLE: float = 0.7   # 70% cue sentences, 30% random


# ── Few-shot classifier ────────────────────────────────────────────────────────

def load_scicite() -> tuple[list[str], list[int], list[str], list[int]]:
    """Load SciCite from its Parquet conversion on the HF Hub.

    intent_classifier.load_scicite() needs the dataset loading script, which the
    installed `datasets` version no longer supports; the Parquet branch has the
    same data and the same label ids (0=method, 1=background, 2=result).

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/scicite", revision="refs/convert/parquet")
    return (ds["train"]["string"], ds["train"]["label"],
            ds["test"]["string"], ds["test"]["label"])


def pick_examples(n_shots: int = 6) -> list[tuple[str, str]]:
    """Choose a fixed set of (sentence, label) examples from the SciCite train split.

    Args:
        n_shots: Total examples; split evenly across the 3 classes (2 each for 6).

    Returns:
        List of (sentence, label_name) pairs; identical on every run (seeded).
    """
    train_texts, train_labels, _, _ = load_scicite()
    rng = random.Random(SEED)                      # local RNG so global state is untouched
    per_class = n_shots // len(LABEL_NAMES)        # examples per class
    shots: list[tuple[str, str]] = []
    for idx, name in enumerate(LABEL_NAMES):
        pool = [t for t, y in zip(train_texts, train_labels) if y == idx]
        shots += [(t, name) for t in rng.sample(pool, per_class)]
    return shots


def build_prompt(sentence: str, shots: list[tuple[str, str]]) -> str:
    """Build the few-shot prompt for one sentence.

    Args:
        sentence: Citation sentence to classify.
        shots:    (sentence, label) examples shown to the model.

    Returns:
        Prompt string that asks for a JSON reply like {"label": "method"}.
    """
    lines = [
        "Classify the intent of the citation in the sentence as one of: "
        + ", ".join(LABEL_NAMES) + ".",
        "Reply with JSON only, e.g. {\"label\": \"method\"}.",
        "",
    ]
    for text, label in shots:
        lines.append(f"Sentence: {text}\nAnswer: {json.dumps({'label': label})}\n")
    lines.append(f"Sentence: {sentence}\nAnswer:")
    return "\n".join(lines)


def parse_label(reply: str) -> str:
    """Extract a valid label from an LLM reply.

    Args:
        reply: Raw model text, ideally {"label": "..."}.

    Returns:
        One of LABEL_NAMES; "background" if nothing valid can be parsed.
    """
    match = re.search(r"\{.*?\}", reply, flags=re.DOTALL)  # first JSON-looking object
    if match:
        try:
            label = str(json.loads(match.group(0)).get("label", "")).strip().lower()
            if label in LABEL_NAMES:
                return label
        except json.JSONDecodeError:
            pass                                    # fall through to the default
    return "background"


def call_with_retry(prompt: str, api_key: str, tries: int = 8) -> str:
    """Call Groq, waiting and retrying when the per-minute token limit (HTTP 429) is hit.

    Args:
        prompt:  Prompt text.
        api_key: Groq API key.
        tries:   Maximum attempts before giving up.

    Returns:
        The model reply text.

    Raises:
        RuntimeError: on a non-429 error, or if all attempts hit the rate limit.
    """
    for attempt in range(tries):
        try:
            return _call_groq(prompt, DEFAULT_GROQ_MODEL, api_key)
        except RuntimeError as err:
            if "429" not in str(err) or attempt == tries - 1:
                raise                                        # real error, or out of retries
            wait = re.search(r"try again in ([\d.]+)s", str(err))  # server tells us how long
            time.sleep(float(wait.group(1)) + 1 if wait else 10)
    raise RuntimeError("unreachable")                        # loop always returns or raises


def classify_fewshot(sentences: list[str], n_shots: int = 6) -> list[str]:
    """Label each sentence as background / method / result with a few-shot LLM.

    Every response is cached on disk by prompt hash, so reruns make no API calls.

    Args:
        sentences: Citation sentences to classify.
        n_shots:   Number of fixed training examples in the prompt.

    Returns:
        List of labels, same order and length as `sentences`.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not set and a call is needed.
    """
    shots = pick_examples(n_shots)
    api_key = os.environ.get(GROQ_API_KEY_ENV, "").strip()   # never hard-coded
    labels: list[str] = []
    for sentence in sentences:
        prompt = build_prompt(sentence, shots)
        key = _cache_key(prompt)
        reply = _load_cache(FEWSHOT_CACHE_DIR, key)          # None if not cached yet
        if reply is None:
            if not api_key:
                raise EnvironmentError(f"{GROQ_API_KEY_ENV} is not set (add it to .env).")
            reply = call_with_retry(prompt, api_key)
            _save_cache(FEWSHOT_CACHE_DIR, key, reply)
        labels.append(parse_label(reply))
    return labels


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_fewshot(n: int = 300) -> dict:
    """Evaluate the few-shot classifier on a seeded sample of SciCite test sentences.

    Args:
        n: Sample size.

    Returns:
        Metrics dict (same keys as results/tfidf_intent.json); also saved to
        results/fewshot_intent.json.
    """
    _, _, test_texts, test_labels = load_scicite()
    idxs = random.Random(SEED).sample(range(len(test_texts)), n)  # fixed sample
    texts = [test_texts[i] for i in idxs]
    gold = [LABEL_NAMES[test_labels[i]] for i in idxs]
    preds = classify_fewshot(texts)

    report = classification_report(
        gold, preds, labels=LABEL_NAMES, output_dict=True, zero_division=0
    )
    metrics = {
        "macro_f1": round(f1_score(gold, preds, labels=LABEL_NAMES, average="macro"), 6),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 6),
                "recall":    round(report[name]["recall"], 6),
                "f1-score":  round(report[name]["f1-score"], 6),
                "support":   int(report[name]["support"]),
            }
            for name in LABEL_NAMES
        },
        "confusion_matrix": confusion_matrix(gold, preds, labels=LABEL_NAMES).tolist(),
        "dataset": "allenai/scicite",
        "model": DEFAULT_GROQ_MODEL,
        "n_samples": n,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FEWSHOT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Macro F1 = {metrics['macro_f1']}  -> {FEWSHOT_METRICS}")
    return metrics


# ── Hand-labelling helpers ─────────────────────────────────────────────────────

def export_label_candidates(n: int = 300) -> None:
    """Sample edges to hand-label as extension / contradiction / other.

    70% of the sample contains a cue phrase (when enough exist); the rest is random.
    Writes data/to_label.csv with columns edge_id, context_sentence, label (empty).
    edge_id is the row position in edges.parquet.

    Args:
        n: Total number of edges to export.
    """
    edges = schema.load_edges(DATA_DIR / "edges.parquet")
    sentences = edges[schema.CONTEXT_SENTENCE].fillna("").str.lower()
    has_cue = sentences.apply(lambda s: any(c in s for c in CUE_PHRASES))

    cue_rows = edges[has_cue]
    n_cue = min(int(n * CUE_FRACTION_OF_SAMPLE), len(cue_rows))
    cue_pick = cue_rows.sample(n=n_cue, random_state=SEED)
    rest = edges.drop(cue_pick.index)                       # avoid duplicate rows
    rand_pick = rest.sample(n=min(n - n_cue, len(rest)), random_state=SEED)

    out = pd.concat([cue_pick, rand_pick]).sample(frac=1, random_state=SEED)  # shuffle
    out = pd.DataFrame({
        "edge_id": out.index,
        schema.CONTEXT_SENTENCE: out[schema.CONTEXT_SENTENCE].values,
        "label": "",
    })
    out.to_csv(TO_LABEL_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(out)} rows ({n_cue} cue-based) -> {TO_LABEL_CSV}")


def load_hand_labels() -> pd.DataFrame:
    """Read the filled data/to_label.csv and save data/hand_labels.parquet.

    Returns:
        DataFrame with edge_id, context_sentence, label.

    Raises:
        ValueError: if any label is empty or not in extension/contradiction/other.
    """
    df = pd.read_csv(TO_LABEL_CSV, dtype={"label": str}, keep_default_na=False)
    df["label"] = df["label"].str.strip().str.lower()        # forgive stray spaces / caps
    bad = df[~df["label"].isin(HAND_LABEL_CHOICES)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have missing/invalid labels, e.g. edge_ids "
                         f"{bad['edge_id'].head(5).tolist()}")
    df.to_parquet(HAND_LABELS_PARQUET, index=False)
    print(f"Saved {len(df)} labels -> {HAND_LABELS_PARQUET}")
    print(df["label"].value_counts().to_string())
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Command-line entry point; each flag runs one function."""
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="classify 5 test sentences")
    p.add_argument("--eval", action="store_true", help="run eval_fewshot (LLM calls)")
    p.add_argument("--export", action="store_true", help="write data/to_label.csv")
    p.add_argument("--load-labels", action="store_true", help="build hand_labels.parquet")
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()

    if args.smoke:
        _, _, texts, labels = load_scicite()
        preds = classify_fewshot(texts[:5])
        for t, g, pr in zip(texts[:5], labels[:5], preds):
            print(f"gold={LABEL_NAMES[g]:<10} pred={pr:<10} {t[:80]}")
    if args.eval:
        eval_fewshot(args.n)
    if args.export:
        export_label_candidates(args.n)
    if args.load_labels:
        load_hand_labels()


if __name__ == "__main__":
    main()
