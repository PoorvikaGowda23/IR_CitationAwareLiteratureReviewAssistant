"""
tests/test_retrieval.py
-----------------------
Unit tests for Step 12 hybrid graph re-ranking retrieval pipeline in retrieval.py:
  - min_max_normalize
  - compute_proximity_to_anchors
  - compute_cocitation_with_anchors
  - retrieve interface contract (keys, types, rank order)
  - weight switchability for ablations
"""

import networkx as nx
import pytest

from retrieval import (
    compute_cocitation_with_anchors,
    compute_proximity_to_anchors,
    min_max_normalize,
    retrieve,
)


def test_min_max_normalize():
    assert min_max_normalize([]) == []
    # Identical values return zeros
    assert min_max_normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
    # Standard min-max normalization
    norm = min_max_normalize([10.0, 20.0, 30.0])
    assert pytest.approx(norm) == [0.0, 0.5, 1.0]


def test_compute_proximity_to_anchors():
    U = nx.Graph()
    # a1 - c1 - c2 - c3 - c4
    U.add_edge("a1", "c1")
    U.add_edge("c1", "c2")
    U.add_edge("c2", "c3")
    U.add_edge("c3", "c4")
    # c5 is isolated

    anchors = ["a1"]
    dist_map = {"a1": nx.single_source_shortest_path_length(U, "a1", cutoff=3)}

    # a1 to itself -> 1 / (1 + 0) = 1.0
    assert compute_proximity_to_anchors("a1", anchors, U, dist_map, max_hops=3) == 1.0
    # c1 is 1 hop -> 1 / (1 + 1) = 0.5
    assert compute_proximity_to_anchors("c1", anchors, U, dist_map, max_hops=3) == 0.5
    # c2 is 2 hops -> 1 / (1 + 2) = 1/3
    assert pytest.approx(compute_proximity_to_anchors("c2", anchors, U, dist_map, max_hops=3)) == 1.0 / 3.0
    # c3 is 3 hops -> 1 / (1 + 3) = 0.25
    assert compute_proximity_to_anchors("c3", anchors, U, dist_map, max_hops=3) == 0.25
    # c4 is 4 hops (> 3 hops cutoff) -> 0.0
    assert compute_proximity_to_anchors("c4", anchors, U, dist_map, max_hops=3) == 0.0
    # c5 is disconnected -> 0.0
    assert compute_proximity_to_anchors("c5", anchors, U, dist_map, max_hops=3) == 0.0


def test_compute_cocitation_with_anchors():
    G = nx.DiGraph()
    # Papers p1 and p2 both cite c and a1
    G.add_edge("p1", "c")
    G.add_edge("p1", "a1")
    G.add_edge("p2", "c")
    G.add_edge("p2", "a1")
    # Paper p3 cites c and a2
    G.add_edge("p3", "c")
    G.add_edge("p3", "a2")

    anchors = ["a1", "a2"]
    # Co-citation with a1 = 2 (p1, p2), with a2 = 1 (p3) -> total = 3
    assert compute_cocitation_with_anchors("c", anchors, G) == 3.0
    # Paper with no in-edges has 0 co-citation
    assert compute_cocitation_with_anchors("p1", anchors, G) == 0.0


def test_retrieve_contract_and_ablation(monkeypatch):
    # Mock stage1_candidates to avoid depending on FAISS/SPECTER in unit tests
    fake_cands = [
        ("p1", 0.95),
        ("p2", 0.90),
        ("p3", 0.85),
        ("p4", 0.80),
        ("p5", 0.75),
        ("p6", 0.70),
    ]
    monkeypatch.setattr("retrieval.stage1_candidates", lambda query, k: fake_cands)

    G = nx.DiGraph()
    for pid, _ in fake_cands:
        G.add_node(pid, pagerank=0.1)
    G.add_edge("p6", "p1")  # link p6 to anchor p1

    # Test contract format
    results = retrieve("fake query", k=3, graph=G)
    assert len(results) == 3
    for rank, item in enumerate(results, start=1):
        assert item["rank"] == rank
        assert isinstance(item["paper_id"], str)
        assert isinstance(item["score"], float)
        assert item["flags"] == []
        assert "components" in item
        comp = item["components"]
        assert set(comp.keys()) == {"embedding", "pagerank", "proximity", "cocitation"}

    # Test ablation: pure embedding (alpha=1, others 0) must match Stage 1 order
    results_pure_emb = retrieve(
        "fake query",
        k=6,
        weights={"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
        graph=G,
        surface_contradictions=False,
    )
    returned_ids = [r["paper_id"] for r in results_pure_emb]
    expected_ids = [p for p, _ in fake_cands]
    assert returned_ids == expected_ids


def test_surface_contradictions(monkeypatch):
    fake_cands = [
        ("p1", 0.95),
        ("p2", 0.90),
        ("p3", 0.85),
    ]
    monkeypatch.setattr("retrieval.stage1_candidates", lambda query, k: fake_cands)

    G = nx.DiGraph()
    for pid, _ in fake_cands:
        G.add_node(pid, pagerank=0.1)

    # p1 contradicts c_out (outgoing edge) with conf 0.85
    G.add_edge("p1", "c_out", intent="contradiction", confidence=0.85)
    # c_in contradicts p2 (incoming edge) with conf 0.75
    G.add_edge("c_in", "p2", intent="contradiction", confidence=0.75)
    # low confidence contradiction edge (should be ignored)
    G.add_edge("p3", "c_low", intent="contradiction", confidence=0.40)

    # 1. With surface_contradictions=True
    results = retrieve("fake query", k=3, graph=G, surface_contradictions=True)
    assert len(results) == 5  # 3 top-k + 2 surfaced contradictions
    assert results[0]["flags"] == []
    assert results[1]["flags"] == []
    assert results[2]["flags"] == []

    # Surfaced papers placed at rank 4 and 5
    contra_items = results[3:]
    assert len(contra_items) == 2
    assert contra_items[0]["rank"] == 4
    assert contra_items[0]["paper_id"] == "c_out"
    assert contra_items[0]["flags"] == ["contradiction", "p1"]

    assert contra_items[1]["rank"] == 5
    assert contra_items[1]["paper_id"] == "c_in"
    assert contra_items[1]["flags"] == ["contradiction", "p2"]

    # 2. With surface_contradictions=False
    results_no_contra = retrieve("fake query", k=3, graph=G, surface_contradictions=False)
    assert len(results_no_contra) == 3

