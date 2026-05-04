from __future__ import annotations

from retrieval.query_classifier import classify_query_type


def route_model(query: str) -> dict:
    qtype = classify_query_type(query)
    use_vision = qtype == "multimodal"
    return {
        "query_type": qtype,
        "routed": "vision_llm" if use_vision else "text_llm",
        "text_llm": {"primary": "llama3.2", "fallback": "llama3"},
        "vision_llm": {"primary": "llava", "fallback": "llama3.2-vision"},
    }
