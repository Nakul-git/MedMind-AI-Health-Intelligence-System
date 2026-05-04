from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.auto_fetch import auto_fetch_for_input
from ingestion.embed_store import build_index
from ingestion.pipeline import run_ingestion_pipeline
from evaluation.evaluate_rag import evaluate_rag
from llm.generator import generate_answer
from medmind.config import DATA_DIR
from retrieval.pipeline import retrieve_top_k_relevant_chunks


def run_query(query: str, top_k: int = 5, metadata_filter: dict | None = None, reasoning_trace: bool = False) -> dict:
    chunks = retrieve_top_k_relevant_chunks(query, top_k=top_k, metadata_filter=metadata_filter)
    answer = generate_answer(query, chunks, reasoning_trace=reasoning_trace)
    return {
        "query": query,
        "ranked_relevant_chunks": chunks,
        **answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MedMind Multimodal RAG")
    parser.add_argument("command", choices=["ingest", "pipeline-ingest", "query", "evaluate", "auto-fetch"], help="Utility command")
    parser.add_argument("--data-dir", default=str(DATA_DIR / "knowledge_base"), help="Input KB directory")
    parser.add_argument("--query", default="", help="User query")
    parser.add_argument("--mode", default="symptoms", choices=["symptoms", "xray", "report", "drug"], help="Auto-fetch mode")
    parser.add_argument("--no-pdf", action="store_true", help="Skip open-access PDF downloads")
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks")
    parser.add_argument("--metadata-filter", default="", help='JSON string, e.g. {"document_type":"pdf"}')
    parser.add_argument("--reasoning-trace", action="store_true", help="Include reasoning_trace_optional")
    args = parser.parse_args()

    if args.command == "ingest":
        index_path = build_index(Path(args.data_dir))
        print(f"Built legacy index at {index_path}")
        return

    if args.command == "pipeline-ingest":
        print(json.dumps(run_ingestion_pipeline(), indent=2, ensure_ascii=False))
        return

    if args.command == "query":
        if not args.query.strip():
            raise SystemExit("--query is required for query")
        metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
        print(
            json.dumps(
                run_query(args.query, top_k=args.top_k, metadata_filter=metadata_filter, reasoning_trace=args.reasoning_trace),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "evaluate":
        print(json.dumps(evaluate_rag(), indent=2, ensure_ascii=False))
        return

    if args.command == "auto-fetch":
        if not args.query.strip():
            raise SystemExit("--query is required for auto-fetch")
        result = auto_fetch_for_input(args.query, mode=args.mode, download_pdfs=not args.no_pdf)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
