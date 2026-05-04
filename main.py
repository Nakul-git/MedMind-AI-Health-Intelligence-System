from __future__ import annotations

import argparse
from pathlib import Path

from ingestion.auto_fetch import auto_fetch_for_input
from ingestion.embed_store import build_index
from medmind.config import DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="MedMind utilities")
    parser.add_argument("command", choices=["ingest", "auto-fetch"], help="Utility command to run")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR / "knowledge_base"),
        help="Directory containing .jsonl/.txt medical knowledge documents",
    )
    parser.add_argument("--query", default="", help="Input query for auto-fetch")
    parser.add_argument("--mode", default="symptoms", choices=["symptoms", "xray", "report", "drug"], help="Auto-fetch mode")
    parser.add_argument("--no-pdf", action="store_true", help="Skip open-access PDF downloads during auto-fetch")
    args = parser.parse_args()

    if args.command == "ingest":
        index_path = build_index(Path(args.data_dir))
        print(f"Built MedMind retrieval index at {index_path}")
    elif args.command == "auto-fetch":
        if not args.query:
            raise SystemExit("--query is required for auto-fetch")
        result = auto_fetch_for_input(args.query, mode=args.mode, download_pdfs=not args.no_pdf)
        print(result)


if __name__ == "__main__":
    main()
