from __future__ import annotations


def build_prompt_final(user_query: str, retrieved_context: list[dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(retrieved_context, start=1):
        context_lines.append(f"[{i}] {chunk.get('title', 'Untitled')} | {chunk.get('text', '')}")
    context = "\n".join(context_lines)
    return (
        "You are a grounded medical RAG assistant.\n"
        "Rules:\n"
        "1) Do not hallucinate.\n"
        "2) Answer only from context.\n"
        "3) Always provide citations [n].\n"
        "4) If not found in context, say: Unknown from provided context.\n\n"
        f"User query: {user_query}\n\n"
        f"Context chunks:\n{context}\n"
    )


def apply_guardrails(answer: str, retrieved_context: list[dict]) -> str:
    text = (answer or "").strip()
    if not text:
        return "Unknown from provided context."

    if "unknown from provided context" not in text.lower():
        has_citation = any(f"[{i}]" in text for i in range(1, len(retrieved_context) + 1))
        if not has_citation:
            text = f"{text}\n\nCitations: [1]"

    for phrase in ["100% certain", "guaranteed cure", "definitely cured", "no doubt"]:
        text = text.replace(phrase, "")
    return text.strip()
