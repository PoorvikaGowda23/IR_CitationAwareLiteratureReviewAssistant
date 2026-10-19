"""
tests/test_metrics.py
---------------------
Unit tests for Step 7 IR evaluation metrics against a hand-computed example.

Hand Calculation Details:
-------------------------
Ranked IDs: ["d1", "d2", "d3", "d4", "d5"]
Relevant:   {"d2", "d4"}

Rank 1: "d1" -> not relevant (rel = 0)
Rank 2: "d2" -> relevant     (rel = 1)
Rank 3: "d3" -> not relevant (rel = 0)
Rank 4: "d4" -> relevant     (rel = 1)
Rank 5: "d5" -> not relevant (rel = 0)

1. Precision@k:
   P@1 = 0 / 1 = 0.0
   P@2 = 1 / 2 = 0.5
   P@3 = 1 / 3 = 0.3333333333333333
   P@4 = 2 / 4 = 0.5
   P@5 = 2 / 5 = 0.4

2. Recall@k (Total relevant = 2):
   R@1 = 0 / 2 = 0.0
   R@2 = 1 / 2 = 0.5
   R@3 = 1 / 2 = 0.5
   R@4 = 2 / 2 = 1.0
   R@5 = 2 / 2 = 1.0

3. Average Precision (AP):
   Relevant items appear at ranks 2 and 4.
   Precision at rank 2 = 1 / 2 = 0.5
   Precision at rank 4 = 2 / 4 = 0.5
   Sum of precision at relevant ranks = 0.5 + 0.5 = 1.0
   AP = (P@2 + P@4) / |relevant| = 1.0 / 2 = 0.50

4. nDCG@k:
   - At k = 5:
     DCG@5:
       rank 1: 0
       rank 2: 1 / log2(2 + 1) = 1 / log2(3) ≈ 0.6309297535714574
       rank 3: 0
       rank 4: 1 / log2(4 + 1) = 1 / log2(5) ≈ 0.43067655807339306
       rank 5: 0
       DCG@5 = 1/log2(3) + 1/log2(5) ≈ 1.0616063116448505

     IDCG@5:
       Ideal ranking places relevant items at ranks 1 and 2:
       rank 1: 1 / log2(1 + 1) = 1 / log2(2) = 1.0
       rank 2: 1 / log2(2 + 1) = 1 / log2(3) ≈ 0.6309297535714574
       IDCG@5 = 1.0 + 1/log2(3) ≈ 1.6309297535714574

     nDCG@5 = DCG@5 / IDCG@5
            = (1/log2(3) + 1/log2(5)) / (1.0 + 1/log2(3))
            ≈ 1.0616063116448505 / 1.6309297535714574
            ≈ 0.6509210086259021

   - At k = 3:
     DCG@3  = 1 / log2(3) ≈ 0.6309297535714574
     IDCG@3 = 1.0 + 1 / log2(3) ≈ 1.6309297535714574
     nDCG@3 = (1/log2(3)) / (1.0 + 1/log2(3)) ≈ 0.38685280723454163
"""

import math
import pytest

from evaluate import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    bootstrap_ci,
    paired_bootstrap,
)

RANKED_IDS = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


def test_precision_at_k():
    assert precision_at_k(RANKED_IDS, RELEVANT, k=1) == pytest.approx(0.0)
    assert precision_at_k(RANKED_IDS, RELEVANT, k=2) == pytest.approx(0.5)
    assert precision_at_k(RANKED_IDS, RELEVANT, k=3) == pytest.approx(1.0 / 3.0)
    assert precision_at_k(RANKED_IDS, RELEVANT, k=4) == pytest.approx(0.5)
    assert precision_at_k(RANKED_IDS, RELEVANT, k=5) == pytest.approx(0.4)


def test_recall_at_k():
    assert recall_at_k(RANKED_IDS, RELEVANT, k=1) == pytest.approx(0.0)
    assert recall_at_k(RANKED_IDS, RELEVANT, k=2) == pytest.approx(0.5)
    assert recall_at_k(RANKED_IDS, RELEVANT, k=3) == pytest.approx(0.5)
    assert recall_at_k(RANKED_IDS, RELEVANT, k=4) == pytest.approx(1.0)
    assert recall_at_k(RANKED_IDS, RELEVANT, k=5) == pytest.approx(1.0)


def test_average_precision():
    # P@2 = 0.5, P@4 = 0.5 -> AP = (0.5 + 0.5) / 2 = 0.5
    assert average_precision(RANKED_IDS, RELEVANT) == pytest.approx(0.5)


def test_ndcg_at_k():
    dcg3 = 1.0 / math.log2(3)
    idcg3 = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_ndcg3 = dcg3 / idcg3
    assert ndcg_at_k(RANKED_IDS, RELEVANT, k=3) == pytest.approx(expected_ndcg3)

    dcg5 = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg5 = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_ndcg5 = dcg5 / idcg5
    assert ndcg_at_k(RANKED_IDS, RELEVANT, k=5) == pytest.approx(expected_ndcg5)


def test_empty_relevant():
    assert precision_at_k(RANKED_IDS, set(), k=3) == 0.0
    assert recall_at_k(RANKED_IDS, set(), k=3) == 0.0
    assert average_precision(RANKED_IDS, set()) == 0.0
    assert ndcg_at_k(RANKED_IDS, set(), k=3) == 0.0


def test_bootstrap_ci():
    scores = [0.4, 0.5, 0.6, 0.5, 0.55]
    ci_low, ci_high = bootstrap_ci(scores, n=500, seed=42)
    assert 0.35 <= ci_low <= ci_high <= 0.65


def test_paired_bootstrap():
    scores_a = [0.8, 0.9, 0.85]
    scores_b = [0.2, 0.3, 0.25]
    p = paired_bootstrap(scores_a, scores_b, n=500, seed=42)
    assert p == 1.0
