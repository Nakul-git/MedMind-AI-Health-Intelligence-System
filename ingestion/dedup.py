from __future__ import annotations


def dedup_documents(docs: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for doc in docs:
        key = (
            str(doc.get("title", "")).strip().lower(),
            str(doc.get("text", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(chunk)
    return out
