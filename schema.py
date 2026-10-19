"""
Column name constants and load/save helpers for all shared Parquet/JSON data files.

Every other module must import column names from here — never use raw strings.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

# ── papers.parquet columns ─────────────────────────────────────────────────────
PAPER_ID        = "paper_id"        # Semantic Scholar corpus ID (string)
TITLE           = "title"           # paper title (string)
ABSTRACT        = "abstract"        # full abstract text (string, may be empty)
YEAR            = "year"            # publication year (int, nullable)
TLDR            = "tldr"            # one-sentence S2 summary (string, nullable)
IS_SURVEY       = "is_survey"       # True if paper is a survey/review (bool)

# Required columns that must be present in papers.parquet
PAPERS_COLS: list[str] = [PAPER_ID, TITLE, ABSTRACT, YEAR, TLDR, IS_SURVEY]

# ── edges.parquet columns ──────────────────────────────────────────────────────
CITING_ID         = "citing_id"         # paper_id of the citing paper
CITED_ID          = "cited_id"          # paper_id of the cited paper
CONTEXT_SENTENCE  = "context_sentence"  # raw sentence in which the citation appears
S2_INTENT         = "s2_intent"         # intent label supplied by S2 API (may be None)
IS_INFLUENTIAL    = "is_influential"    # S2 "influential citation" flag (bool)
YEAR_GAP          = "year_gap"          # citing year − cited year (int)

# Required columns that must be present in edges.parquet
EDGES_COLS: list[str] = [
    CITING_ID, CITED_ID, CONTEXT_SENTENCE,
    S2_INTENT, IS_INFLUENTIAL, YEAR_GAP,
]

# ── edges_pred.parquet columns ─────────────────────────────────────────────────
# All EDGES_COLS plus the two classifier outputs below
PREDICTED_INTENT   = "predicted_intent"    # our model's intent label (string)
INTENT_CONFIDENCE  = "intent_confidence"   # softmax probability of predicted label (float)

# Required columns that must be present in edges_pred.parquet
EDGES_PRED_COLS: list[str] = EDGES_COLS + [PREDICTED_INTENT, INTENT_CONFIDENCE]

# ── ground_truth.json keys ─────────────────────────────────────────────────────
GT_QUERY_ID          = "query_id"           # unique identifier for this query
GT_QUERY_TEXT        = "query_text"         # natural-language query string
GT_SPLIT             = "split"              # "train" | "val" | "test"
GT_RELEVANT_PAPER_IDS = "relevant_paper_ids"  # list[str] of relevant paper_ids

# Required keys that must be present in every ground-truth record
GT_KEYS: list[str] = [GT_QUERY_ID, GT_QUERY_TEXT, GT_SPLIT, GT_RELEVANT_PAPER_IDS]


# ── Internal helper ────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Raise ValueError if any required column is missing from df.

    Args:
        df:       DataFrame to validate.
        required: List of column names that must exist.
        source:   Human-readable label (e.g. file path) used in the error message.
    """
    missing = [c for c in required if c not in df.columns]  # find absent columns
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _check_gt_keys(records: list[dict[str, Any]], source: str) -> None:
    """Raise ValueError if any ground-truth record is missing a required key.

    Args:
        records: List of dicts loaded from ground_truth.json.
        source:  Human-readable label used in the error message.
    """
    for i, rec in enumerate(records):
        missing = [k for k in GT_KEYS if k not in rec]  # check each record
        if missing:
            raise ValueError(
                f"{source}: record[{i}] is missing required keys: {missing}\n"
                f"Found keys: {list(rec.keys())}"
            )


# ── papers.parquet ─────────────────────────────────────────────────────────────

def load_papers(path: Path | str) -> pd.DataFrame:
    """Load papers.parquet and validate that all required columns are present.

    Args:
        path: Path to the Parquet file.

    Returns:
        DataFrame with at least the columns defined in PAPERS_COLS.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if a required column is missing.
    """
    path = Path(path)
    df = pd.read_parquet(path)              # read the Parquet file into a DataFrame
    _check_columns(df, PAPERS_COLS, str(path))  # validate columns before returning
    return df


def save_papers(df: pd.DataFrame, path: Path | str) -> None:
    """Validate and save a papers DataFrame to Parquet.

    Args:
        df:   DataFrame containing at least the columns in PAPERS_COLS.
        path: Destination Parquet file path (parent dirs must exist).

    Raises:
        ValueError: if a required column is missing.
    """
    path = Path(path)
    _check_columns(df, PAPERS_COLS, "papers DataFrame")  # guard before writing
    df.to_parquet(path, index=False)        # write without the pandas index


# ── edges.parquet ──────────────────────────────────────────────────────────────

def load_edges(path: Path | str) -> pd.DataFrame:
    """Load edges.parquet and validate that all required columns are present.

    Args:
        path: Path to the Parquet file.

    Returns:
        DataFrame with at least the columns defined in EDGES_COLS.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if a required column is missing.
    """
    path = Path(path)
    df = pd.read_parquet(path)
    _check_columns(df, EDGES_COLS, str(path))
    return df


def save_edges(df: pd.DataFrame, path: Path | str) -> None:
    """Validate and save an edges DataFrame to Parquet.

    Args:
        df:   DataFrame containing at least the columns in EDGES_COLS.
        path: Destination Parquet file path.

    Raises:
        ValueError: if a required column is missing.
    """
    path = Path(path)
    _check_columns(df, EDGES_COLS, "edges DataFrame")
    df.to_parquet(path, index=False)


# ── edges_pred.parquet ─────────────────────────────────────────────────────────

def load_edges_pred(path: Path | str) -> pd.DataFrame:
    """Load edges_pred.parquet (edges + classifier predictions) and validate columns.

    Args:
        path: Path to the Parquet file.

    Returns:
        DataFrame with at least the columns defined in EDGES_PRED_COLS.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if a required column is missing.
    """
    path = Path(path)
    df = pd.read_parquet(path)
    _check_columns(df, EDGES_PRED_COLS, str(path))
    return df


def save_edges_pred(df: pd.DataFrame, path: Path | str) -> None:
    """Validate and save an edges-with-predictions DataFrame to Parquet.

    Args:
        df:   DataFrame containing at least the columns in EDGES_PRED_COLS.
        path: Destination Parquet file path.

    Raises:
        ValueError: if a required column is missing.
    """
    path = Path(path)
    _check_columns(df, EDGES_PRED_COLS, "edges_pred DataFrame")
    df.to_parquet(path, index=False)


# ── ground_truth.json ──────────────────────────────────────────────────────────

def load_ground_truth(path: Path | str) -> list[dict[str, Any]]:
    """Load ground_truth.json and validate that every record has the required keys.

    Args:
        path: Path to the JSON file (must be a JSON array of objects).

    Returns:
        List of dicts, each with keys defined in GT_KEYS.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if a record is missing a required key.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)              # parse the JSON array
    _check_gt_keys(records, str(path))      # validate every record
    return records


def save_ground_truth(records: list[dict[str, Any]], path: Path | str) -> None:
    """Validate and save ground-truth records to a JSON file.

    Args:
        records: List of dicts, each with keys defined in GT_KEYS.
        path:    Destination JSON file path.

    Raises:
        ValueError: if a record is missing a required key.
    """
    path = Path(path)
    _check_gt_keys(records, "ground_truth records")   # guard before writing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)     # pretty-print for readability
