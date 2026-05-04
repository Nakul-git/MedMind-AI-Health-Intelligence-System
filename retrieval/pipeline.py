from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from medmind.ollama_client import embed_texts
from retrieval.advanced_ops import llm_generated_variations, mmr_select, reciprocal_rank_fusion
from retrieval.query_classifier import classify_query_type

CHROMA_DIR = Path("db/chroma_db")
COLLECTION_NAME = "multimodal_rag_chunks"


def _load_embedding_cache() -> dict[str, list[float]]:
    cache_path = Path("data/vector_database/embeddings.json")
    meta_path = Path("data/structured_chunks/metadata_enriched_docs.jsonl")
    if not cache_path.exists() or not meta_path.exists():
        return {}
    embeddings = json.loads(cache_path.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: dict[str, list[float]] = {}
    for i, chunk in enumerate(chunks):
        if i < len(embeddings):
            out[str(chunk.get("chunk_id"))] = embeddings[i]
    return out


def _chroma_dense_search(query: str, top_n: int = 30) -> list[dict]:
    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(COLLECTION_NAME)
        query_embedding = embed_texts([query])
        if query_embedding:
            res = collection.query(query_embeddings=query_embedding, n_results=top_n, include=["documents", "metadatas", "distances"])
        else:
            res = collection.query(query_texts=[query], n_results=top_n, include=["documents", "metadatas", "distances"])

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        emb_map = _load_embedding_cache()

        out = []
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "chunk_id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": float(dists[i]) if i < len(dists) else 1.0,
                    "embedding": emb_map.get(str(doc_id)),
                }
            )
        return out
    except Exception:
        return []


def _bm25_sparse_search(query: str, docs: list[dict], top_n: int = 30) -> list[tuple[str, float]]:
    if not docs:
        return []
    corpus = [d.get("text", "").lower().split() for d in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(
        [(str(docs[i].get("chunk_id", f"idx_{i}")), float(scores[i])) for i in range(len(docs))],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def _apply_metadata_filter(docs: list[dict], metadata_filter: dict | None) -> list[dict]:
    if not metadata_filter:
        return docs
    out = []
    for d in docs:
        md = d.get("metadata", {})
        if all(md.get(k) == v for k, v in metadata_filter.items()):
            out.append(d)
    return out


def _cross_encoder_minilm_rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = model.predict(pairs)
        rows = []
        for d, s in zip(docs, scores):
            item = dict(d)
            item["rerank_score"] = float(s)
            rows.append(item)
        return sorted(rows, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    except Exception:
        q = set(query.lower().split())
        rows = []
        for d in docs:
            txt = d.get("text", "").lower()
            overlap = len(q.intersection(set(txt.split())))
            item = dict(d)
            item["rerank_score"] = float(d.get("score", 0.0)) + overlap * 0.02
            rows.append(item)
        return sorted(rows, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def retrieve_top_k_relevant_chunks(query: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[dict]:
    qtype = classify_query_type(query)
    queries = llm_generated_variations(query, qtype)

    dense_map: dict[str, dict] = {}
    dense_rankings: list[list[tuple[str, float]]] = []
    sparse_rankings: list[list[tuple[str, float]]] = []

    for q in queries:
        dense_docs = _apply_metadata_filter(_chroma_dense_search(q, top_n=max(20, top_k * 8)), metadata_filter)
        q_emb = embed_texts([q])
        if q_emb:
            dense_docs = mmr_select(q_emb[0], dense_docs, top_k=max(15, top_k * 4), lambda_mult=0.65)

        for d in dense_docs:
            dense_map[str(d.get("chunk_id"))] = d

        dense_rankings.append([(str(d.get("chunk_id")), 1.0 / (1.0 + float(d.get("distance", 1.0)))) for d in dense_docs])
        sparse_rankings.append(_bm25_sparse_search(q, dense_docs, top_n=max(15, top_k * 4)))

    fused = reciprocal_rank_fusion(dense_rankings + sparse_rankings)

    fused_docs = []
    for chunk_id, score in fused:
        base = dense_map.get(chunk_id)
        if not base:
            continue
        fused_docs.append(
            {
                "id": chunk_id,
                "title": base.get("metadata", {}).get("title", "Untitled"),
                "text": base.get("text", ""),
                "source": base.get("metadata", {}).get("source", ""),
                "metadata": base.get("metadata", {}),
                "score": float(score),
                "query_type": qtype,
            }
        )

    return _cross_encoder_minilm_rerank(query, fused_docs, top_k=top_k)
