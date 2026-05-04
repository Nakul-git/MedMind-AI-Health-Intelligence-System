from __future__ import annotations


MEDICAL_PRIORITY_TERMS = {
    "urgent": 0.08,
    "warning": 0.07,
    "guideline": 0.05,
    "drug": 0.04,
    "lab": 0.04,
}


def rerank(query: str, docs: list[dict]) -> list[dict]:
    lower_query = query.lower()
    query_terms = set(lower_query.split())
    wants_safety = any(term in lower_query for term in ["urgent", "warning", "danger", "red flag", "emergency"])
    wants_drug = any(term in lower_query for term in ["drug", "medicine", "dose", "side effect", "warning"])
    wants_lab = any(term in lower_query for term in ["lab", "report", "blood", "cbc", "glucose", "platelet"])
    wants_symptom = any(term in lower_query for term in ["fever", "pain", "cough", "headache", "rash", "vomiting"])
    reranked = []
    for doc in docs:
        title = doc.get("title", "").lower()
        text = f"{title} {doc.get('text', '')}".lower()
        source_type = str(doc.get("source_type", "")).lower()
        bonus = 0.0
        if wants_safety:
            bonus += sum(MEDICAL_PRIORITY_TERMS[term] for term in ["urgent", "warning", "guideline"] if term in text)
        if wants_drug:
            bonus += MEDICAL_PRIORITY_TERMS["drug"] if "drug" in text else 0.0
        if wants_lab:
            bonus += MEDICAL_PRIORITY_TERMS["lab"] if "lab" in text else 0.0
        if wants_drug and source_type in {"drug", "drug_data", "medication", "drug_reference"}:
            bonus += 0.18
        if wants_lab and source_type in {"open_access_pdf", "pubmed_summary", "europe_pmc_open_access", "uploaded_report"}:
            bonus += 0.1
        if wants_symptom and source_type in {"pubmed_summary", "europe_pmc_open_access", "open_access_pdf"}:
            bonus += 0.08
        if wants_drug and any(bad in text for bad in ["twitter", "social media", "nlp classification"]) and "paracetamol" in lower_query:
            bonus -= 0.12
        overlap = len(query_terms & set(text.split())) * 0.01
        exact_title_bonus = sum(0.15 for term in query_terms if len(term) > 5 and term in title)
        updated = dict(doc)
        updated["score"] = float(doc.get("score", 0.0)) + bonus + overlap + exact_title_bonus
        reranked.append(updated)
    return sorted(reranked, key=lambda item: item["score"], reverse=True)
