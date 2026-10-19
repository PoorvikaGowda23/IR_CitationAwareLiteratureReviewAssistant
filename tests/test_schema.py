"""
Round-trip smoke tests for schema.py loaders and savers.

Each test builds a minimal fake DataFrame / list-of-dicts, saves it to a
pytest `tmp_path` directory, loads it back, and checks that the result is
identical to what was saved. Tests also verify that missing columns/keys
raise the correct error.
"""

import json
import pytest
import pandas as pd

import schema


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_papers() -> pd.DataFrame:
    """Return a one-row DataFrame with every required papers column."""
    return pd.DataFrame([{
        schema.PAPER_ID:  "p001",
        schema.TITLE:     "A Fake Paper",
        schema.ABSTRACT:  "Abstract text.",
        schema.YEAR:      2023,
        schema.TLDR:      "It does stuff.",
        schema.IS_SURVEY: False,
    }])


def _make_edges() -> pd.DataFrame:
    """Return a one-row DataFrame with every required edges column."""
    return pd.DataFrame([{
        schema.CITING_ID:        "p002",
        schema.CITED_ID:         "p001",
        schema.CONTEXT_SENTENCE: "We follow [1].",
        schema.S2_INTENT:        "method",
        schema.IS_INFLUENTIAL:   True,
        schema.YEAR_GAP:         2,
    }])


def _make_edges_pred() -> pd.DataFrame:
    """Return a one-row DataFrame with every required edges_pred column."""
    base = _make_edges()
    # Add the two classifier output columns on top of the base edge columns
    base[schema.PREDICTED_INTENT]  = "method"
    base[schema.INTENT_CONFIDENCE] = 0.95
    return base


def _make_ground_truth() -> list[dict]:
    """Return a one-item ground-truth list with every required key."""
    return [{
        schema.GT_QUERY_ID:           "q001",
        schema.GT_QUERY_TEXT:         "What is BERT?",
        schema.GT_SPLIT:              "test",
        schema.GT_RELEVANT_PAPER_IDS: ["p001", "p002"],
    }]


# ── papers round-trip ──────────────────────────────────────────────────────────

def test_papers_roundtrip(tmp_path):
    """save_papers then load_papers returns the original DataFrame."""
    df_orig = _make_papers()
    dest = tmp_path / "papers.parquet"
    schema.save_papers(df_orig, dest)           # write to disk
    df_loaded = schema.load_papers(dest)        # read back
    # Check every required column value is unchanged
    pd.testing.assert_frame_equal(df_orig, df_loaded, check_like=True)


def test_papers_missing_column(tmp_path):
    """load_papers raises ValueError when a required column is absent."""
    dest = tmp_path / "papers_bad.parquet"
    # Drop 'abstract' to simulate a malformed file
    bad_df = _make_papers().drop(columns=[schema.ABSTRACT])
    bad_df.to_parquet(dest, index=False)        # write without validation
    with pytest.raises(ValueError, match="abstract"):
        schema.load_papers(dest)


def test_save_papers_missing_column(tmp_path):
    """save_papers raises ValueError before writing if a column is missing."""
    bad_df = _make_papers().drop(columns=[schema.IS_SURVEY])
    with pytest.raises(ValueError, match="is_survey"):
        schema.save_papers(bad_df, tmp_path / "papers.parquet")


# ── edges round-trip ───────────────────────────────────────────────────────────

def test_edges_roundtrip(tmp_path):
    """save_edges then load_edges returns the original DataFrame."""
    df_orig = _make_edges()
    dest = tmp_path / "edges.parquet"
    schema.save_edges(df_orig, dest)
    df_loaded = schema.load_edges(dest)
    pd.testing.assert_frame_equal(df_orig, df_loaded, check_like=True)


def test_edges_missing_column(tmp_path):
    """load_edges raises ValueError when a required column is absent."""
    dest = tmp_path / "edges_bad.parquet"
    bad_df = _make_edges().drop(columns=[schema.YEAR_GAP])
    bad_df.to_parquet(dest, index=False)
    with pytest.raises(ValueError, match="year_gap"):
        schema.load_edges(dest)


# ── edges_pred round-trip ──────────────────────────────────────────────────────

def test_edges_pred_roundtrip(tmp_path):
    """save_edges_pred then load_edges_pred returns the original DataFrame."""
    df_orig = _make_edges_pred()
    dest = tmp_path / "edges_pred.parquet"
    schema.save_edges_pred(df_orig, dest)
    df_loaded = schema.load_edges_pred(dest)
    pd.testing.assert_frame_equal(df_orig, df_loaded, check_like=True)


def test_edges_pred_missing_classifier_column(tmp_path):
    """load_edges_pred raises ValueError when a classifier column is absent."""
    dest = tmp_path / "edges_pred_bad.parquet"
    # Edges without the predicted_intent column look like plain edges
    bad_df = _make_edges()
    bad_df.to_parquet(dest, index=False)
    with pytest.raises(ValueError, match="predicted_intent"):
        schema.load_edges_pred(dest)


# ── ground_truth round-trip ────────────────────────────────────────────────────

def test_ground_truth_roundtrip(tmp_path):
    """save_ground_truth then load_ground_truth returns the original records."""
    records_orig = _make_ground_truth()
    dest = tmp_path / "ground_truth.json"
    schema.save_ground_truth(records_orig, dest)
    records_loaded = schema.load_ground_truth(dest)
    assert records_loaded == records_orig           # exact equality for plain dicts


def test_ground_truth_missing_key(tmp_path):
    """load_ground_truth raises ValueError when a required key is absent."""
    dest = tmp_path / "gt_bad.json"
    # Write a record that is missing 'split'
    bad_records = [{
        schema.GT_QUERY_ID:           "q001",
        schema.GT_QUERY_TEXT:         "test query",
        schema.GT_RELEVANT_PAPER_IDS: ["p001"],
        # deliberately omitting GT_SPLIT
    }]
    with open(dest, "w") as f:
        json.dump(bad_records, f)
    with pytest.raises(ValueError, match="split"):
        schema.load_ground_truth(dest)


def test_save_ground_truth_missing_key(tmp_path):
    """save_ground_truth raises ValueError before writing if a key is missing."""
    bad_records = [{
        schema.GT_QUERY_ID:  "q001",
        schema.GT_QUERY_TEXT: "test query",
        schema.GT_SPLIT:      "test",
        # deliberately omitting GT_RELEVANT_PAPER_IDS
    }]
    with pytest.raises(ValueError, match="relevant_paper_ids"):
        schema.save_ground_truth(bad_records, tmp_path / "gt.json")
