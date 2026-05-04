from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.dedup import dedup_chunks, dedup_documents
from ingestion.parse_medical_pdf import extract_pdf_text
from medmind.config import DATA_DIR
from medmind.ollama_client import embed_texts, generate_vision

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = Path("db/chroma_db")


@dataclass
class IngestionStats:
    documents: int = 0
    chunks: int = 0
    embeddings: int = 0


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def _recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        part = text[start:end].strip()
        if part:
            out.append(part)
        if end >= len(text):
            break
        start += step
    return out


def _partition_pdf_unstructured(path: Path) -> tuple[str, list[dict[str, Any]]]:
    text_blocks: list[str] = []
    tables: list[dict[str, Any]] = []
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore

        elements = partition_pdf(
            filename=str(path),
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
        )
        for el in elements:
            t = str(getattr(el, "text", "") or "").strip()
            if t:
                text_blocks.append(t)
            meta = getattr(el, "metadata", None)
            if meta and getattr(meta, "text_as_html", None):
                tables.append({"source": str(path), "html": meta.text_as_html})
    except Exception:
        fallback = extract_pdf_text(path)
        if fallback:
            text_blocks.append(fallback)
    return ("\n".join(text_blocks), tables)


def _collect_raw_documents() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = {
        "pdf_documents": DATA_DIR / "pdf_documents",
        "text_files": DATA_DIR / "text_files",
        "images": DATA_DIR / "images",
        "scanned_documents": DATA_DIR / "scanned_documents",
    }
    aliases = {
        "pdf_documents": [DATA_DIR / "downloaded_data"],
        "text_files": [DATA_DIR / "extracted_text"],
        "images": [DATA_DIR / "extracted_pdf_images_figures"],
        "scanned_documents": [],
    }
    for p in roots.values():
        p.mkdir(parents=True, exist_ok=True)

    docs: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    for root in [roots["pdf_documents"], *aliases["pdf_documents"], roots["scanned_documents"]]:
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            text, table_rows = _partition_pdf_unstructured(path)
            docs.append(
                {
                    "id": path.stem,
                    "title": path.stem.replace("_", " "),
                    "source": str(path),
                    "document_type": "scanned_document" if root == roots["scanned_documents"] else "pdf",
                    "text": text,
                    "metadata": {
                        "document_type": "scanned_document" if root == roots["scanned_documents"] else "pdf",
                        "source": "local",
                        "date": path.stat().st_mtime,
                        "path": str(path),
                    },
                }
            )
            tables.extend(table_rows)

    for root in [roots["text_files"], *aliases["text_files"]]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".txt", ".md", ".json", ".jsonl"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(
                {
                    "id": path.stem,
                    "title": path.stem.replace("_", " "),
                    "source": str(path),
                    "document_type": "text",
                    "text": text,
                    "metadata": {
                        "document_type": "text",
                        "source": "local",
                        "date": path.stat().st_mtime,
                        "path": str(path),
                    },
                }
            )

    for root in [roots["images"], *aliases["images"]]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            caption, _ = generate_vision(
                "Describe medical image findings and visual context for retrieval.",
                path,
            )
            docs.append(
                {
                    "id": path.stem,
                    "title": path.stem.replace("_", " "),
                    "source": str(path),
                    "document_type": "image",
                    "text": caption or f"Image: {path.name}",
                    "metadata": {
                        "document_type": "image",
                        "source": "local",
                        "date": path.stat().st_mtime,
                        "path": str(path),
                    },
                }
            )

    return dedup_documents(docs), tables


def _persist_to_chroma(chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> str:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection("multimodal_rag_chunks")
        ids = [c["chunk_id"] for c in chunks]
        existing = collection.get(ids=ids)
        existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
        if existing_ids:
            collection.delete(ids=existing_ids)
        docs = [c["text"] for c in chunks]
        metas = [c["metadata"] | {"title": c.get("title", ""), "source": c.get("source", "")} for c in chunks]
        if embeddings and len(embeddings) == len(chunks):
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        else:
            collection.add(ids=ids, documents=docs, metadatas=metas)
        return str(CHROMA_DIR)
    except Exception:
        return "chroma_unavailable"


def run_ingestion_pipeline(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> dict[str, Any]:
    stats = IngestionStats()
    docs, tables = _collect_raw_documents()

    docs = dedup_documents([{**d, "text": _normalize_text(str(d.get("text", "")))} for d in docs])
    stats.documents = len(docs)

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        for i, piece in enumerate(_recursive_chunk(doc["text"], chunk_size=chunk_size, overlap=chunk_overlap)):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_{i}",
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "text": piece,
                    "source": doc.get("source", ""),
                    "metadata": {
                        **doc.get("metadata", {}),
                        "relevance_score": 0.0,
                    },
                }
            )

    chunks = dedup_chunks(chunks)
    stats.chunks = len(chunks)

    payloads = [f"{c.get('title', '')}\n{c.get('text', '')}" for c in chunks]
    embeddings = embed_texts(payloads) or []
    stats.embeddings = len(embeddings)

    for d in [DATA_DIR / "extracted_text", DATA_DIR / "extracted_images", DATA_DIR / "table_structures", DATA_DIR / "metadata", DATA_DIR / "structured_chunks", DATA_DIR / "vector_database"]:
        d.mkdir(parents=True, exist_ok=True)

    (DATA_DIR / "extracted_text" / "extracted_text.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs if d.get("document_type") != "image"),
        encoding="utf-8",
    )
    (DATA_DIR / "extracted_images" / "extracted_images.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs if d.get("document_type") == "image"),
        encoding="utf-8",
    )
    (DATA_DIR / "table_structures" / "table_structures.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tables),
        encoding="utf-8",
    )
    (DATA_DIR / "metadata" / "metadata.jsonl").write_text(
        "\n".join(json.dumps({"id": d["id"], **d.get("metadata", {})}, ensure_ascii=False) for d in docs),
        encoding="utf-8",
    )
    (DATA_DIR / "structured_chunks" / "metadata_enriched_docs.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )
    (DATA_DIR / "vector_database" / "embeddings.json").write_text(json.dumps(embeddings), encoding="utf-8")

    chroma_path = _persist_to_chroma(chunks, embeddings)

    framework_status = {"langchain": False, "llamaindex": False}
    try:
        import langchain  # type: ignore # noqa: F401

        framework_status["langchain"] = True
    except Exception:
        pass
    try:
        import llama_index  # type: ignore # noqa: F401

        framework_status["llamaindex"] = True
    except Exception:
        pass

    return {
        "multimodal_chunks": stats.chunks,
        "embeddings": stats.embeddings,
        "metadata_enriched_docs": stats.documents,
        "table_structures": len(tables),
        "vector_db": "chroma",
        "persist_directory": chroma_path,
        "frameworks": framework_status,
    }
