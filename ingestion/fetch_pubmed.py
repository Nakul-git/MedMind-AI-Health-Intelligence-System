from __future__ import annotations

import json
from pathlib import Path

import requests

from medmind.config import DATA_DIR


def fetch_pubmed_summaries(query: str, limit: int = 5) -> Path:
    """Fetch PubMed summaries through NCBI E-utilities and save JSONL documents."""
    out_dir = DATA_DIR / "medical_papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"pubmed_{query.lower().replace(' ', '_')}.jsonl"

    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": limit},
        timeout=20,
    )
    search.raise_for_status()
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        output.write_text("", encoding="utf-8")
        return output

    summary = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        timeout=20,
    )
    summary.raise_for_status()
    data = summary.json().get("result", {})

    with output.open("w", encoding="utf-8") as handle:
        for pubmed_id in ids:
            item = data.get(pubmed_id, {})
            title = item.get("title", f"PubMed {pubmed_id}")
            text = f"{title}. Journal: {item.get('fulljournalname', '')}. Published: {item.get('pubdate', '')}."
            handle.write(
                json.dumps(
                    {
                        "id": f"pubmed_{pubmed_id}",
                        "title": title,
                        "source_type": "pubmed",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                        "text": text,
                    }
                )
                + "\n"
            )
    return output


if __name__ == "__main__":
    print(fetch_pubmed_summaries("fever headache differential diagnosis"))

