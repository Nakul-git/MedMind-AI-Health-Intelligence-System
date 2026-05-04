from __future__ import annotations


def classify_query_type(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["compare", "analyze", "why", "impact", "tradeoff"]):
        return "analytical"
    if any(k in q for k in ["image", "xray", "x-ray", "scan", "ct", "mri", "figure"]):
        return "multimodal"
    if any(k in q for k in ["summarize", "summary", "overview", "tl;dr"]):
        return "summarization"
    return "factual"
