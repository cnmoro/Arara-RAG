"""IR metrics, implemented to match trec_eval / pytrec_eval conventions.

MTEB-BR reports ``ndcg_at_10`` as the main retrieval score, so these functions
are written to agree with ``pytrec_eval`` rather than with a re-derivation.
``bench/validate_metrics.py`` checks that agreement empirically.
"""

from __future__ import annotations

import math


def _dcg(gains: list[float], k: int) -> float:
    """Discounted cumulative gain with *linear* gain.

    trec_eval exposes two NDCG variants: ``ndcg`` (exponential gain,
    ``2^rel - 1``) and ``ndcg_cut`` (linear gain). pytrec_eval -- and therefore
    MTEB -- uses ``ndcg_cut``, so linear gain is what we must reproduce. This
    was verified empirically against pytrec_eval rather than assumed.
    """
    total = 0.0
    for i, gain in enumerate(gains[:k], start=1):
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain with graded relevance."""
    if not qrels:
        return 0.0
    gains = [float(qrels.get(doc_id, 0)) for doc_id in ranked[:k]]
    ideal = _dcg(sorted((float(v) for v in qrels.values()), reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return _dcg(gains, k) / ideal


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Fraction of judged-relevant documents found in the top ``k``."""
    if not qrels:
        return 0.0
    relevant = {d for d, v in qrels.items() if v > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top ``k``."""
    relevant = {d for d, v in qrels.items() if v > 0}
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (10, 100),
) -> dict[str, float]:
    """Aggregate metrics over queries. Queries with no qrels are skipped."""
    qids = [q for q in rankings if qrels.get(q)]
    if not qids:
        return {}
    out: dict[str, float] = {}
    for k in k_values:
        out[f"ndcg_at_{k}"] = sum(ndcg_at_k(rankings[q], qrels[q], k) for q in qids) / len(qids)
        out[f"recall_at_{k}"] = sum(recall_at_k(rankings[q], qrels[q], k) for q in qids) / len(qids)
        out[f"mrr_at_{k}"] = sum(mrr_at_k(rankings[q], qrels[q], k) for q in qids) / len(qids)
    out["n_queries"] = float(len(qids))
    return out
