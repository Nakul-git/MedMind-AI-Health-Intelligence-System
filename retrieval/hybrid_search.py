from __future__ import annotations

import math
from collections import Counter

from ingestion.embed_store import load_index, tokenize
from medmind.ollama_client import embed_texts
from retrieval.medical_reranker import rerank
from retrieval.multi_query import expand_query
from retrieval.rrf import reciprocal_rank_fusion


def _lexical_rank(query: str, docs: list[dict]) -> list[tuple[int, float]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []
    doc_tokens = [tokenize(doc["text"] + " " + doc.get("title", "")) for doc in docs]
    doc_freq = Counter(term for tokens in doc_tokens for term in set(tokens))
    total_docs = len(docs)
    scores = []
    for idx, tokens in enumerate(doc_tokens):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if counts[term]:
                idf = math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1
                score += counts[term] * idf
        scores.append((idx, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _rank_ollama_embedding(query_vector: list[float], index: dict) -> list[tuple[int, float]]:
    query_norm = math.sqrt(sum(float(value) * float(value) for value in query_vector)) or 1.0
    scores = []
    for idx, vector in enumerate(index["ollama_embeddings"]):
        dot = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
        scores.append((idx, dot / (query_norm * index["ollama_norms"][idx])))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _tfidf_vector_rank(query: str, index: dict) -> list[tuple[int, float]]:
    query_counts = Counter(tokenize(query))
    total_docs = index["total_docs"]
    doc_freq = index["doc_freq"]
    query_vector = {
        term: (1 + math.log(count)) * (math.log((total_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1)
        for term, count in query_counts.items()
    }
    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0
    scores = []
    for idx, vector in enumerate(index["vectors"]):
        dot = sum(weight * vector.get(term, 0.0) for term, weight in query_vector.items())
        scores.append((idx, dot / (query_norm * index["norms"][idx])))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _vector_rank(query: str, index: dict) -> list[tuple[int, float]]:
    if index.get("ollama_embeddings") and index.get("ollama_norms"):
        query_embedding = embed_texts([query], timeout=30)
        if query_embedding:
            return _rank_ollama_embedding(query_embedding[0], index)
    return _tfidf_vector_rank(query, index)


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    index = load_index()
    docs = index["documents"]
    rankings: list[list[tuple[int, float]]] = []
    expanded_queries = expand_query(query)

    for expanded in expanded_queries:
        rankings.append(_lexical_rank(expanded, docs)[: top_k * 3])

    if index.get("ollama_embeddings") and index.get("ollama_norms"):
        query_embeddings = embed_texts([query], timeout=45)
        if query_embeddings:
            for query_vector in query_embeddings:
                rankings.append(_rank_ollama_embedding(query_vector, index)[: top_k * 3])
        else:
            for expanded in expanded_queries:
                rankings.append(_tfidf_vector_rank(expanded, index)[: top_k * 3])
    else:
        for expanded in expanded_queries:
            rankings.append(_tfidf_vector_rank(expanded, index)[: top_k * 3])

    fused = reciprocal_rank_fusion(rankings)
    selected = []
    for doc_idx, score in fused[: top_k * 2]:
        doc = dict(docs[doc_idx])
        doc["score"] = round(float(score), 4)
        selected.append(doc)
    return rerank(query, selected)[:top_k]
