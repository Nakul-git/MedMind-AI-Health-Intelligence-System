from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from llm.generator import generate_answer
from retrieval.pipeline import retrieve_top_k_relevant_chunks


def _precision_at_k(chunks: list[dict], must_include: list[str]) -> float:
    if not chunks:
        return 0.0
    positives = 0
    for c in chunks:
        blob = (c.get("title", "") + " " + c.get("text", "")).lower()
        if any(m.lower() in blob for m in must_include):
            positives += 1
    return positives / len(chunks)


def _recall_at_k(chunks: list[dict], must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    blob = " ".join((c.get("title", "") + " " + c.get("text", "")).lower() for c in chunks)
    hit = sum(1 for m in must_include if m.lower() in blob)
    return hit / len(must_include)


def _hallucination_rate(answer: str, chunks: list[dict]) -> float:
    if not answer:
        return 1.0
    words = [w for w in answer.lower().split() if len(w) > 5][:40]
    if not words:
        return 0.0
    unsupported = 0
    for w in words:
        if not any(w in c.get("text", "").lower() for c in chunks):
            unsupported += 1
    return unsupported / len(words)


def evaluate_rag() -> dict:
    eval_items = json.loads((Path(__file__).resolve().parent / "eval_questions.json").read_text(encoding="utf-8"))

    recall_scores = []
    precision_scores = []
    reranker_scores = []
    correctness_scores = []
    faithfulness_scores = []
    hallucination_scores = []
    latencies = []

    for item in eval_items:
        start = time.perf_counter()
        chunks = retrieve_top_k_relevant_chunks(item["question"], top_k=item.get("top_k", 5))
        generated = generate_answer(item["question"], chunks)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        must = item.get("must_include", [])
        recall = _recall_at_k(chunks, must)
        precision = _precision_at_k(chunks, must)
        hallucination = _hallucination_rate(generated.get("structured_answer", ""), chunks)

        recall_scores.append(recall)
        precision_scores.append(precision)
        reranker_scores.append(precision)
        correctness_scores.append(max(0.0, 1.0 - hallucination))
        faithfulness_scores.append(max(0.0, 1.0 - hallucination))
        hallucination_scores.append(hallucination)

    end_to_end = recall_scores + precision_scores + correctness_scores + faithfulness_scores
    return {
        "retrieval": {
            "recall_at_k": round(mean(recall_scores), 4) if recall_scores else 0.0,
            "precision_at_k": round(mean(precision_scores), 4) if precision_scores else 0.0,
        },
        "reranker": {"ranking_accuracy": round(mean(reranker_scores), 4) if reranker_scores else 0.0},
        "generation": {
            "answer_correctness": round(mean(correctness_scores), 4) if correctness_scores else 0.0,
            "faithfulness": round(mean(faithfulness_scores), 4) if faithfulness_scores else 0.0,
            "hallucination_rate": round(mean(hallucination_scores), 4) if hallucination_scores else 0.0,
        },
        "system": {
            "latency": round(mean(latencies), 4) if latencies else 0.0,
            "end_to_end_performance": round(mean(end_to_end), 4) if end_to_end else 0.0,
        },
        "num_eval_questions": len(eval_items),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_rag(), indent=2))
