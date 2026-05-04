from __future__ import annotations

import math
from collections import defaultdict

from medmind.ollama_client import generate_text


def _fallback_variations(query: str, query_type: str) -> list[str]:
    variants = [query]
    if query_type == "analytical":
        variants.append(f"clinical analysis {query}")
    if query_type == "multimodal":
        variants.append(f"visual findings {query}")
    if query_type == "summarization":
        variants.append(f"key summary points {query}")
    variants.append(f"evidence based {query}")
    return list(dict.fromkeys(variants))[:5]


def llm_generated_variations(query: str, query_type: str) -> list[str]:
    prompt = (
        "Generate 4 concise medical search query variations as newline list. "
        f"Query type: {query_type}. Original: {query}"
    )
    text, _ = generate_text(prompt, timeout=20)
    if not text:
        return _fallback_variations(query, query_type)
    lines = [line.strip(" -\t\"'") for line in text.splitlines() if line.strip()]
    candidates = [query] + [line for line in lines if len(line) > 5]
    return list(dict.fromkeys(candidates))[:5]


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def mmr_select(query_embedding: list[float], docs: list[dict], top_k: int, lambda_mult: float = 0.6) -> list[dict]:
    def cos(a: list[float], b: list[float]) -> float:
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    selected: list[dict] = []
    candidates = docs[:]
    while candidates and len(selected) < top_k:
        best = None
        best_score = -1e9
        for cand in candidates:
            emb = cand.get("embedding")
            sim_query = cos(query_embedding, emb) if emb else 0.0
            sim_selected = 0.0
            if selected and emb:
                sims = [cos(emb, s.get("embedding", [])) for s in selected if s.get("embedding")]
                sim_selected = max(sims) if sims else 0.0
            score = lambda_mult * sim_query - (1 - lambda_mult) * sim_selected
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
    return selected
