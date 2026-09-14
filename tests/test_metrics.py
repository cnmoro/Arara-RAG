"""Metric tests, including the exact convention MTEB relies on."""

from __future__ import annotations

import math

import pytest

from bench.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_uses_linear_gain() -> None:
    """trec_eval's ``ndcg_cut`` uses linear gain, not 2^rel - 1.

    Verified against pytrec_eval in bench/validate_metrics.py; this test pins
    the convention so a future 'fix' to exponential gain fails loudly.
    """
    qrels = {"a": 2, "b": 1}
    ranked = ["a", "x", "b"]
    dcg = 2 / math.log2(2) + 0 + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, qrels, 10) == pytest.approx(dcg / idcg)
    # exponential gain would give a different, wrong answer
    assert ndcg_at_k(ranked, qrels, 10) != pytest.approx(0.963940, abs=1e-5)


def test_ndcg_perfect_and_zero() -> None:
    qrels = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b"], qrels, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], qrels, 10) == 0.0
    assert ndcg_at_k([], qrels, 10) == 0.0


def test_recall_and_mrr() -> None:
    qrels = {"a": 1, "b": 1, "c": 1, "d": 1}
    assert recall_at_k(["a", "b"], qrels, 10) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], qrels, 2) == pytest.approx(0.5)
    assert recall_at_k(["x", "a"], qrels, 10) == pytest.approx(0.25)
    assert mrr_at_k(["x", "a", "b"], qrels, 10) == pytest.approx(0.5)
    assert mrr_at_k(["a"], qrels, 10) == pytest.approx(1.0)
    assert mrr_at_k(["x", "y"], qrels, 10) == 0.0


def test_mrr_respects_cutoff() -> None:
    qrels = {"a": 1}
    assert mrr_at_k(["x"] * 10 + ["a"], qrels, 10) == 0.0
    assert mrr_at_k(["x"] * 10 + ["a"], qrels, 11) == pytest.approx(1 / 11)


def test_evaluate_skips_queries_without_qrels() -> None:
    rankings = {"q1": ["a"], "q2": ["b"], "q3": ["c"]}
    qrels = {"q1": {"a": 1}, "q2": {}}
    out = evaluate(rankings, qrels, k_values=(10,))
    assert out["n_queries"] == 1.0
    assert out["ndcg_at_10"] == pytest.approx(1.0)
