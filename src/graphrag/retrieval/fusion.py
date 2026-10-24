"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(chunk) = sum over rankings of 1 / (k + rank)

Why RRF instead of a weighted sum of scores: vector distances and hop counts are on
incomparable scales, and RRF uses only ranks, so no calibration is needed. A chunk
found by BOTH retrievers gets two contributions, so agreement is rewarded. k=60 is
the standard constant; it flattens the gap between rank 1 and rank 2.
"""

from __future__ import annotations

from collections import defaultdict


def rrf(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings.values():
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)
