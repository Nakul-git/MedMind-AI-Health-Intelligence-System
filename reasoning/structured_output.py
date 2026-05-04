from __future__ import annotations

from medmind.models import EvidenceSource


def sources_from_docs(docs: list[dict]) -> list[EvidenceSource]:
    sources: list[EvidenceSource] = []
    for doc in docs:
        text = doc.get("text", "")
        snippet = text[:260] + ("..." if len(text) > 260 else "")
        sources.append(
            EvidenceSource(
                title=doc.get("title", "Untitled source"),
                source_type=doc.get("source_type", "unknown"),
                url=doc.get("url") or None,
                snippet=snippet,
                score=float(doc.get("score", 0.0)),
            )
        )
    return sources

