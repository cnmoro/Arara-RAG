"""Rank fusion.

Reciprocal Rank Fusion is used rather than score normalization because the two
retrievers produce scores on incomparable scales (cosine in [-1, 1] versus an
unbounded BM25 sum), and RRF needs no per-query calibration to combine them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Fuse several ranked id lists into one score map.

    Args:
        rankings: each element is an ordered iterable of ids, best first.
        k: the RRF damping constant. 60 is the value from the original paper.
        weights: optional per-ranking multipliers.

    Returns:
        Mapping from id to fused score. Ids absent from every ranking are
        absent from the result.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused


def rank_from_scores(scores: dict[int, float], top_k: int | None = None) -> list[tuple[int, float]]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return ordered
