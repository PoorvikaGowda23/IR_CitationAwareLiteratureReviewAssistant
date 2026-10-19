"""
tests/test_graph_builder.py
---------------------------
Unit tests for Step 11 graph builder functions:
  - build_graph (weights, untyped, duplicate collapse, ext_intent override, survey exclusion)
  - compute_pagerank (weighted PageRank properties)
  - cocitation_matrix (hand-computed example)
  - coupling_matrix (hand-computed example)
  - louvain_communities (seeding and partitioning)
  - save_graph / load_graph (roundtrip with node attributes)
"""

import pickle
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from config import INTENT_WEIGHTS, SEED
from graph_builder import (
    annotate_graph,
    build_graph,
    cocitation_matrix,
    compute_pagerank,
    coupling_matrix,
    louvain_communities,
    load_graph,
    save_graph,
)


def test_build_graph_weights_and_collapse():
    # p1 cites p2 twice: first as background (0.3), second as method (1.0)
    # Highest weight (1.0, method) should be kept.
    edges_df = pd.DataFrame([
        {"citing_id": "p1", "cited_id": "p2", "predicted_intent": "background"},
        {"citing_id": "p1", "cited_id": "p2", "predicted_intent": "method"},
        {"citing_id": "p2", "cited_id": "p3", "predicted_intent": "background"},
    ])

    G = build_graph(edges_df, untyped=False, survey_ids=set())
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    # Collapsed to method with weight 1.0
    assert G["p1"]["p2"]["intent"] == "method"
    assert G["p1"]["p2"]["weight"] == INTENT_WEIGHTS["method"]
    assert G["p2"]["p3"]["intent"] == "background"
    assert G["p2"]["p3"]["weight"] == INTENT_WEIGHTS["background"]


def test_build_graph_untyped():
    edges_df = pd.DataFrame([
        {"citing_id": "p1", "cited_id": "p2", "predicted_intent": "background"},
    ])
    G = build_graph(edges_df, untyped=True, survey_ids=set())
    assert G["p1"]["p2"]["weight"] == 1.0


def test_build_graph_ext_intent_override():
    # If ext_intent is extension with confidence > 0.6, it overrides predicted_intent
    edges_df = pd.DataFrame([
        {
            "citing_id": "p1", "cited_id": "p2",
            "predicted_intent": "background",
            "ext_intent": "extension",
            "ext_confidence": 0.85,
        },
        {
            "citing_id": "p2", "cited_id": "p3",
            "predicted_intent": "method",
            "ext_intent": "contradiction",
            "ext_confidence": 0.50,  # below 0.6 threshold -> keep predicted_intent
        },
    ])
    G = build_graph(edges_df, untyped=False, survey_ids=set())
    assert G["p1"]["p2"]["intent"] == "extension"
    assert G["p1"]["p2"]["weight"] == INTENT_WEIGHTS["extension"]
    assert G["p2"]["p3"]["intent"] == "method"


def test_survey_exclusion_and_assertion():
    survey_id = "survey_001"
    edges_df = pd.DataFrame([
        {"citing_id": "p1", "cited_id": survey_id, "predicted_intent": "background"},
        {"citing_id": survey_id, "cited_id": "p2", "predicted_intent": "method"},
        {"citing_id": "p1", "cited_id": "p2", "predicted_intent": "method"},
    ])
    G = build_graph(edges_df, untyped=False, survey_ids={survey_id})
    assert survey_id not in G.nodes
    assert G.has_edge("p1", "p2")


def test_compute_pagerank():
    G = nx.DiGraph()
    G.add_edge("p1", "p2", weight=1.0)
    G.add_edge("p3", "p2", weight=1.0)
    pr = compute_pagerank(G, alpha=0.85)
    # p2 has all in-edges, should have highest PageRank
    assert pr["p2"] > pr["p1"]
    assert pr["p2"] > pr["p3"]
    assert pytest.approx(sum(pr.values()), rel=1e-5) == 1.0


def test_cocitation_and_coupling_hand_computed():
    """
    Hand calculation:
    Nodes: [p0, p1, p2, p3]
    Edges:
      p0 -> p1, p0 -> p2  (p0 cites both p1 and p2 -> co-citation for (p1, p2))
      p1 -> p3, p2 -> p3  (both p1 and p2 cite p3 -> bibliographic coupling for (p1, p2))
    """
    G = nx.DiGraph()
    G.add_edge("p0", "p1")
    G.add_edge("p0", "p2")
    G.add_edge("p1", "p3")
    G.add_edge("p2", "p3")

    nodes = ["p0", "p1", "p2", "p3"]
    C = cocitation_matrix(G, nodelist=nodes)
    B = coupling_matrix(G, nodelist=nodes)

    # Co-citation: p0 cites p1 and p2, so entry for (p1, p2) is 1
    idx1, idx2 = nodes.index("p1"), nodes.index("p2")
    assert C[idx1, idx2] == 1
    assert C[idx2, idx1] == 1

    # Bibliographic coupling: p1 and p2 both cite p3, so entry for (p1, p2) is 1
    assert B[idx1, idx2] == 1
    assert B[idx2, idx1] == 1

    # p0 does not share references with p3
    idx0, idx3 = nodes.index("p0"), nodes.index("p3")
    assert B[idx0, idx3] == 0


def test_louvain_communities():
    G = nx.DiGraph()
    # Cluster 1: p1, p2, p3
    G.add_edge("p1", "p2", weight=1.0)
    G.add_edge("p2", "p3", weight=1.0)
    G.add_edge("p3", "p1", weight=1.0)
    # Cluster 2: q1, q2, q3
    G.add_edge("q1", "q2", weight=1.0)
    G.add_edge("q2", "q3", weight=1.0)
    G.add_edge("q3", "q1", weight=1.0)

    comms = louvain_communities(G, seed=SEED)
    assert len(comms) == 2
    # Verify deterministic partitioning
    all_nodes = set().union(*comms)
    assert all_nodes == {"p1", "p2", "p3", "q1", "q2", "q3"}


def test_save_and_load_roundtrip(tmp_path):
    G = nx.DiGraph()
    G.add_edge("p1", "p2", weight=1.0, intent="method")
    annotate_graph(G, seed=SEED)

    save_path = tmp_path / "test_graph.gpickle"
    save_graph(G, path=save_path)

    loaded_G = load_graph(save_path)
    assert set(loaded_G.nodes) == {"p1", "p2"}
    assert "pagerank" in loaded_G.nodes["p1"]
    assert "community" in loaded_G.nodes["p1"]
    assert loaded_G["p1"]["p2"]["intent"] == "method"
