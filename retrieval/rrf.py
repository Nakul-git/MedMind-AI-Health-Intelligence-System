from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)

