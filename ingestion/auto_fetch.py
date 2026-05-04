from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests

from ingestion.embed_store import build_index
from ingestion.parse_medical_pdf import extract_pdf_text
from medmind.config import DATA_DIR


AUTO_JSONL = DATA_DIR / "knowledge_base" / "auto_ingested.jsonl"
PDF_DIR = DATA_DIR / "medical_papers" / "pdfs"


def _slug(text: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:limit] or "medical_query"


def _doc_id(source: str, key: str) -> str:
    digest = hashlib.sha1(f"{source}:{key}".encode("utf-8")).hexdigest()[:14]
    return f"auto_{source}_{digest}"


def _load_existing_ids() -> set[str]:
    if not AUTO_JSONL.exists():
        return set()
    ids = set()
    for line in AUTO_JSONL.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            ids.add(json.loads(line).get("id", ""))
        except json.JSONDecodeError:
            continue
    return ids


def _append_docs(docs: list[dict[str, Any]]) -> int:
    AUTO_JSONL.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_ids()
    new_docs = [doc for doc in docs if doc["id"] not in existing and doc.get("text")]
    if not new_docs:
        return 0
    with AUTO_JSONL.open("a", encoding="utf-8") as handle:
        for doc in new_docs:
            handle.write(json.dumps(doc, ensure_ascii=True) + "\n")
    return len(new_docs)


def _pubmed_docs(query: str, limit: int) -> list[dict[str, Any]]:
    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": limit},
        timeout=20,
    )
    search.raise_for_status()
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    summary = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        timeout=20,
    )
    summary.raise_for_status()
    data = summary.json().get("result", {})
    docs = []
    for pubmed_id in ids:
        item = data.get(pubmed_id, {})
        title = item.get("title") or f"PubMed {pubmed_id}"
        text = (
            f"{title}. Journal: {item.get('fulljournalname', '')}. "
            f"Published: {item.get('pubdate', '')}. "
            f"Authors: {', '.join(author.get('name', '') for author in item.get('authors', [])[:4])}."
        )
        docs.append(
            {
                "id": _doc_id("pubmed", pubmed_id),
                "title": title,
                "source_type": "pubmed_summary",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                "text": text,
                "query": query,
            }
        )
    return docs


def _europe_pmc_docs(query: str, limit: int, download_pdfs: bool) -> tuple[list[dict[str, Any]], list[str]]:
    response = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={
            "query": f"({query}) OPEN_ACCESS:y",
            "format": "json",
            "pageSize": limit,
            "resultType": "core",
        },
        timeout=25,
    )
    response.raise_for_status()
    results = response.json().get("resultList", {}).get("result", [])
    docs: list[dict[str, Any]] = []
    pdfs: list[str] = []

    for item in results:
        pmcid = item.get("pmcid") or item.get("id") or item.get("doi") or item.get("title", "")
        title = item.get("title") or f"Europe PMC {pmcid}"
        abstract = item.get("abstractText") or ""
        journal = item.get("journalTitle") or ""
        year = item.get("pubYear") or ""
        url = f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}"
        docs.append(
            {
                "id": _doc_id("epmc", str(pmcid)),
                "title": title,
                "source_type": "europe_pmc_open_access",
                "url": url,
                "text": f"{title}. {abstract} Journal: {journal}. Published: {year}.",
                "query": query,
            }
        )

        if download_pdfs:
            for link in item.get("fullTextUrlList", {}).get("fullTextUrl", []) or []:
                if (link.get("documentStyle") or "").lower() == "pdf" or str(link.get("url", "")).lower().endswith(".pdf"):
                    pdf_doc = _download_pdf(link["url"], title, query)
                    if pdf_doc:
                        docs.append(pdf_doc)
                        pdfs.append(str(pdf_doc.get("local_pdf", "")))
                    break
    return docs, pdfs


def _download_pdf(url: str, title: str, query: str) -> dict[str, Any] | None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{_slug(title)}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}.pdf"
    path = PDF_DIR / name
    try:
        response = requests.get(url, timeout=35, headers={"User-Agent": "MedMind educational research prototype"})
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return None
        path.write_bytes(response.content)
        text = extract_pdf_text(path)
    except Exception:
        return None

    if len(text) < 300:
        return None
    return {
        "id": _doc_id("pdf", url),
        "title": title,
        "source_type": "open_access_pdf",
        "url": url,
        "text": text[:20000],
        "query": query,
        "local_pdf": str(path),
    }


def build_medical_query(user_input: str, mode: str = "symptoms") -> str:
    base = user_input.strip()
    if mode == "xray":
        return f'chest xray radiology findings differential diagnosis {base}'
    if mode == "report":
        return f'clinical laboratory interpretation guideline abnormal blood report {base}'
    if mode == "drug":
        return f'drug safety dosage adverse effects guideline {base}'
    return f'clinical guideline differential diagnosis symptoms {base}'


def auto_fetch_for_input(
    user_input: str,
    mode: str = "symptoms",
    pubmed_limit: int = 5,
    epmc_limit: int = 4,
    download_pdfs: bool = True,
    rebuild: bool = True,
) -> dict[str, Any]:
    query = build_medical_query(user_input, mode=mode)
    docs: list[dict[str, Any]] = []
    pdfs: list[str] = []
    errors: list[str] = []

    try:
        docs.extend(_pubmed_docs(query, pubmed_limit))
    except Exception as exc:
        errors.append(f"PubMed fetch failed: {exc}")

    try:
        epmc_docs, epmc_pdfs = _europe_pmc_docs(query, epmc_limit, download_pdfs=download_pdfs)
        docs.extend(epmc_docs)
        pdfs.extend(epmc_pdfs)
    except Exception as exc:
        errors.append(f"Europe PMC fetch failed: {exc}")

    added = _append_docs(docs)
    index_path = str(build_index()) if rebuild and added else None
    return {
        "query": query,
        "mode": mode,
        "documents_found": len(docs),
        "documents_added": added,
        "pdfs_downloaded": [path for path in pdfs if path],
        "index_rebuilt": bool(index_path),
        "index_path": index_path,
        "errors": errors,
        "auto_jsonl": str(AUTO_JSONL),
    }


if __name__ == "__main__":
    print(json.dumps(auto_fetch_for_input("fever headache body pain for 3 days"), indent=2))
