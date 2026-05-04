from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()

