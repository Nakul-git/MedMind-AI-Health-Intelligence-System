from __future__ import annotations

from llm.prompt import apply_guardrails, build_prompt_final
from llm.router import route_model
from medmind.ollama_client import generate_text


def generate_answer(user_query: str, retrieved_context: list[dict], reasoning_trace: bool = False) -> dict:
    route = route_model(user_query)
    prompt = build_prompt_final(user_query, retrieved_context)

    raw, model = generate_text(prompt, timeout=90)
    if not raw:
        raw = "Unknown from provided context."

    final_answer = apply_guardrails(raw, retrieved_context)
    citations = [
        {
            "id": i,
            "title": c.get("title", "Untitled"),
            "source": c.get("source") or c.get("metadata", {}).get("path", ""),
        }
        for i, c in enumerate(retrieved_context, start=1)
    ]

    payload = {
        "structured_answer": final_answer,
        "citations": citations,
        "model_routing": route,
        "model_used": model or ("llava" if route["routed"] == "vision_llm" else "llama3.2"),
    }
    if reasoning_trace:
        payload["reasoning_trace_optional"] = {
            "query_type": route["query_type"],
            "retrieved_chunks": len(retrieved_context),
            "guardrails_applied": True,
        }
    return payload
