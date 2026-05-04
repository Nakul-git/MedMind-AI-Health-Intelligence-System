from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

from medmind.ollama_client import embed_texts
from medmind.config import DATA_DIR, EMBEDDINGS_DIR

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


def _read_txt(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": path.stem,
            "title": path.stem.replace("_", " ").title(),
            "source_type": "local_text",
            "url": "",
            "text": path.read_text(encoding="utf-8", errors="ignore"),
        }
    ]


def load_documents(data_dir: Path | None = None) -> list[dict[str, Any]]:
    roots = [
        data_dir or DATA_DIR / "knowledge_base",
        DATA_DIR / "drug_data",
    ]
    documents: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() == ".jsonl":
                documents.extend(_read_jsonl(path))
            elif path.suffix.lower() == ".txt":
                documents.extend(_read_txt(path))
    return [doc for doc in documents if doc.get("text")]


def build_index(data_dir: Path | None = None) -> Path:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents(data_dir)
    if not docs:
        raise RuntimeError("No MedMind documents found to ingest.")

    corpus = [f"{doc.get('title', '')}\n{doc['text']}" for doc in docs]
    tokenized = [tokenize(text) for text in corpus]
    doc_freq = Counter(term for tokens in tokenized for term in set(tokens))
    total_docs = len(docs)
    vectors: list[dict[str, float]] = []
    norms: list[float] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        vector = {
            term: (1 + math.log(count)) * (math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
        vectors.append(vector)
        norms.append(norm)
    ollama_embeddings = embed_texts(corpus, timeout=120)
    ollama_norms = None
    if ollama_embeddings:
        ollama_norms = [
            math.sqrt(sum(float(value) * float(value) for value in embedding)) or 1.0
            for embedding in ollama_embeddings
        ]

    payload = {
        "documents": docs,
        "doc_freq": doc_freq,
        "vectors": vectors,
        "norms": norms,
        "total_docs": total_docs,
        "embedding_model": "mxbai-embed-large" if ollama_embeddings else "tfidf-fallback",
        "ollama_embeddings": ollama_embeddings,
        "ollama_norms": ollama_norms,
    }

    index_path = EMBEDDINGS_DIR / "medmind_tfidf.pkl"
    with index_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return index_path


def load_index() -> dict[str, Any]:
    index_path = EMBEDDINGS_DIR / "medmind_tfidf.pkl"
    if not index_path.exists():
        build_index()
    with index_path.open("rb") as handle:
        return pickle.load(handle)
