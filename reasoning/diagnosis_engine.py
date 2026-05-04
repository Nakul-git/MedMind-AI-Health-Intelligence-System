from __future__ import annotations

import json
import re
from typing import Any

from medmind.config import DISCLAIMER
from medmind.models import HealthInsight, PossibleCondition, ReportInsight, LabFinding
from medmind.ollama_client import generate_text
from reasoning.risk_engine import assess_risk, medical_help_message
from reasoning.structured_output import sources_from_docs
from retrieval.hybrid_search import hybrid_search


CONDITION_RULES = [
    ("Viral fever", ["fever", "headache", "body pain", "fatigue"]),
    ("Influenza or flu-like illness", ["fever", "cough", "chills", "body pain", "sore throat"]),
    ("Dengue-like illness", ["high fever", "rash", "pain behind eyes", "joint pain", "platelets"]),
    ("Respiratory infection", ["cough", "sore throat", "runny nose", "breathing"]),
    ("Gastrointestinal infection", ["diarrhea", "vomiting", "abdominal pain", "nausea"]),
    ("Cardiac or emergency chest-pain condition", ["chest pain", "shortness of breath", "sweating", "jaw pain"]),
]

SECTION_PATTERNS = {
    "history": [r"\bhistory\b", r"\bhpi\b", r"\bpast medical history\b"],
    "exam": [r"\bexam\b", r"\bphysical examination\b", r"\bneurological examination\b"],
    "findings": [r"\bfindings\b", r"\bobservation\b", r"\bassessment\b"],
    "impression": [r"\bimpression\b", r"\bconclusion\b", r"\bopinion\b"],
    "plan": [r"\bplan\b", r"\brecommendation\b", r"\bmanagement\b"],
}

NARRATIVE_FLAG_PATTERNS = {
    "airway_or_ventilation": [r"\breintubat", r"\bintubat", r"\bventilat"],
    "neurologic_red_flag": [r"\bseizure\b", r"\baltered mental status\b", r"\bunresponsive\b", r"\bfocal deficit\b"],
    "hemodynamic_instability": [r"\bhypotens", r"\bshock\b", r"\btachycardia\b"],
    "bleeding_or_stroke_risk": [r"\bhemorrhag", r"\bintracranial\b", r"\bbleed", r"\bstroke\b"],
    "infection_risk": [r"\bmening", r"\bencephal", r"\bsepsis\b", r"\bhigh fever\b"],
}

LAB_PATTERNS = {
    "hemoglobin": (r"(?:hemoglobin|hb)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(g/dl)?", 12.0, 16.0),
    "white blood cells": (r"(?:white blood cells|wbc)\s*[:\-]?\s*([0-9,]+(?:\.[0-9]+)?)", 4000.0, 11000.0),
    "platelets": (r"(?:platelets|platelet count)\s*[:\-]?\s*([0-9,]+(?:\.[0-9]+)?)", 150000.0, 450000.0),
    "fasting glucose": (r"(?:fasting glucose|glucose)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", 70.0, 99.0),
    "hba1c": (r"(?:hba1c|a1c)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", 4.0, 5.6),
}

LAB_KEYWORD_RULES = {
    "fasting glucose": {"keywords": ["glucose", "fasting", "plasma", "fbs"], "low": 70.0, "high": 99.0},
    "hba1c": {"keywords": ["hba1c", "a1c", "glycated hemoglobin"], "low": 4.0, "high": 5.6},
    "white blood cells": {"keywords": ["wbc", "white blood cells"], "low": 4000.0, "high": 11000.0},
    "platelets": {"keywords": ["platelets", "platelet count"], "low": 150000.0, "high": 450000.0},
    "hemoglobin": {"keywords": ["hemoglobin", "hb"], "low": 12.0, "high": 16.0},
}

DRUG_SYNONYMS = {
    "paracetamol": ["acetaminophen"],
    "acetaminophen": ["paracetamol"],
    "ibuprofen": ["advil", "brufen"],
}


def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower()))


def _relevance_overlap_score(query: str, doc: dict) -> float:
    q_tokens = _normalize_tokens(query)
    if not q_tokens:
        return 0.0
    text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
    d_tokens = _normalize_tokens(text)
    overlap = len(q_tokens.intersection(d_tokens))
    return overlap / max(1, len(q_tokens))


def _expand_drug_query_terms(question: str) -> set[str]:
    tokens = _normalize_tokens(question)
    expanded = set(tokens)
    for token in list(tokens):
        for syn in DRUG_SYNONYMS.get(token, []):
            expanded.add(syn)
    return expanded


def _strict_filter_docs(query: str, docs: list[dict], mode: str) -> list[dict]:
    filtered: list[dict] = []
    if mode == "drug":
        drug_terms = _expand_drug_query_terms(query)
        for doc in docs:
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            overlap = len(drug_terms.intersection(_normalize_tokens(text)))
            source_type = str(doc.get("source_type", "")).lower()
            is_drugish = any(
                marker in text
                for marker in ["dose", "dosage", "side effect", "adverse", "warning", "contraindication", "medicine", "drug"]
            )
            source_ok = source_type in {"drug", "drug_data", "medication", "drug_reference", "pubmed_summary", "europe_pmc_open_access", "open_access_pdf", "uploaded_report"}
            if overlap >= 1 and is_drugish and source_ok:
                doc2 = dict(doc)
                doc2["strict_score"] = _relevance_overlap_score(query, doc)
                filtered.append(doc2)
        filtered.sort(key=lambda x: (x.get("strict_score", 0.0), x.get("score", 0.0)), reverse=True)
        return filtered[:6]

    if mode == "report":
        for doc in docs:
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            is_report_context = any(k in text for k in ["lab", "report", "clinical", "guideline", "glucose", "cbc", "platelet", "hba1c"])
            if is_report_context or _relevance_overlap_score(query, doc) >= 0.08:
                doc2 = dict(doc)
                doc2["strict_score"] = _relevance_overlap_score(query, doc)
                filtered.append(doc2)
        filtered.sort(key=lambda x: (x.get("strict_score", 0.0), x.get("score", 0.0)), reverse=True)
        return filtered[:6]

    # symptoms/general
    for doc in docs:
        score = _relevance_overlap_score(query, doc)
        if score >= 0.07:
            doc2 = dict(doc)
            doc2["strict_score"] = score
            filtered.append(doc2)
    filtered.sort(key=lambda x: (x.get("strict_score", 0.0), x.get("score", 0.0)), reverse=True)
    return filtered[:6]


def _extract_primary_drug_terms(question: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", question.lower())
    stop = {
        "what", "are", "is", "the", "of", "and", "for", "with", "in", "on", "about",
        "side", "effects", "warnings", "dose", "dosage", "drug", "medicine", "tablet",
        "capsule", "syrup", "uses", "information",
    }
    meds = [t for t in tokens if t not in stop]
    if not meds:
        return set()
    primary = meds[0]
    terms = {primary}
    for syn in DRUG_SYNONYMS.get(primary, []):
        terms.add(syn)
    return terms


def _drug_doc_mentions_requested_drug(question: str, doc: dict) -> bool:
    terms = _extract_primary_drug_terms(question)
    if not terms:
        return False
    text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
    tokens = _normalize_tokens(text)
    return any(term in tokens for term in terms)


def _split_sentences(text: str) -> list[str]:
    lines = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [line.strip() for line in lines if line.strip()]


def _safe_float(value: str) -> float | None:
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_value_by_keywords(text: str, keywords: list[str]) -> str | None:
    lower = text.lower()
    # Search around each keyword and pick first plausible numeric value in range.
    for keyword in keywords:
        idx = lower.find(keyword)
        if idx == -1:
            continue
        window = text[max(0, idx - 20) : idx + 220]
        numbers = re.findall(r"\b([0-9]{1,6}(?:\.[0-9]{1,3})?)\b", window)
        if not numbers:
            continue
        for token in numbers:
            num = _safe_float(token)
            if num is None:
                continue
            # Reject obvious non-result tokens from dates/times and tiny IDs.
            if num <= 0:
                continue
            if len(token) >= 2:
                return token
    return None


def _extract_sections(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    sections: dict[str, list[str]] = {}
    current = "general"
    sections[current] = []

    def _match_section(line: str) -> str | None:
        lower = line.lower()
        for name, patterns in SECTION_PATTERNS.items():
            if any(re.search(pattern, lower) for pattern in patterns):
                return name
        return None

    for line in lines:
        matched = _match_section(line)
        if matched:
            current = matched
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {k: " ".join(v).strip() for k, v in sections.items() if v}


def _narrative_findings(report_text: str) -> list[LabFinding]:
    findings: list[LabFinding] = []
    sentences = _split_sentences(report_text)
    lower = report_text.lower()

    for label, patterns in NARRATIVE_FLAG_PATTERNS.items():
        hits = [p for p in patterns if re.search(p, lower)]
        if not hits:
            continue
        evidence = []
        for sentence in sentences:
            if any(re.search(p, sentence.lower()) for p in patterns):
                evidence.append(sentence)
            if len(evidence) >= 2:
                break
        findings.append(
            LabFinding(
                test=label.replace("_", " ").title(),
                value="Narrative signal",
                status="high" if "red_flag" in label or "instability" in label else "unknown",
                explanation=" | ".join(evidence) if evidence else "Narrative clinical signal detected.",
            )
        )

    return findings


def _llm_report_summary(report_text: str, sections: dict[str, str], docs: list[dict]) -> str:
    context = "\n\n".join(f"{doc['title']}: {doc['text'][:500]}" for doc in docs[:4])
    section_text = "\n".join(f"{k}: {v[:700]}" for k, v in sections.items())
    prompt = (
        "Summarize this medical report in 5 concise bullets for clinician review. "
        "Focus on key findings, risk-relevant events, and immediate follow-up needs. "
        "If uncertain, say uncertain. Do not diagnose.\n\n"
        f"Report:\n{report_text[:5000]}\n\n"
        f"Sections:\n{section_text}\n\n"
        f"Retrieved context:\n{context}"
    )
    llm_text, _ = generate_text(prompt, timeout=45)
    return llm_text.strip() if llm_text else ""


def _condition_candidates(text: str) -> list[PossibleCondition]:
    lower = text.lower()
    candidates: list[PossibleCondition] = []
    for name, signals in CONDITION_RULES:
        hits = [signal for signal in signals if signal in lower]
        if not hits:
            continue
        likelihood = "high" if len(hits) >= 3 else "medium" if len(hits) == 2 else "low"
        candidates.append(PossibleCondition(name=name, likelihood=likelihood, evidence=hits))
    if not candidates:
        candidates.append(
            PossibleCondition(
                name="Non-specific symptoms requiring clinical context",
                likelihood="low",
                evidence=["insufficient symptom pattern"],
            )
        )
    return candidates[:5]


def analyze_symptoms(symptoms: str) -> HealthInsight:
    docs = _strict_filter_docs(symptoms, hybrid_search(symptoms, top_k=10), mode="symptoms")
    if not docs:
        docs = [
            {
                "title": "Insufficient relevant evidence",
                "source_type": "system",
                "text": "No high-relevance clinical context was retrieved for this symptom query.",
                "score": 0.0,
            }
        ]
    risk_level, risk_hits = assess_risk(symptoms)
    conditions = _condition_candidates(symptoms)
    confidence = min(0.86, 0.45 + (0.08 * len(docs)) + (0.04 * len(conditions)))

    context = "\n\n".join(f"{doc['title']}: {doc['text']}" for doc in docs)
    prompt = (
        "You are a cautious medical education assistant. Do not diagnose. "
        "Summarize likely educational possibilities, risk, and next actions.\n"
        f"Symptoms: {symptoms}\nClinical context:\n{context}\nDisclaimer: {DISCLAIMER}"
    )
    llm_text, model_used = generate_text(
        prompt,
        system="You are MedMind, a cautious medical education assistant. Never diagnose. Recommend emergency care for red flags.",
        timeout=45,
    )
    explanation = llm_text or (
        "The symptom pattern was compared with retrieved clinical and drug references. "
        "The listed possibilities are educational differentials, not diagnoses."
    )

    recommendation = (
        "Rest, hydrate, track temperature and symptom changes, and avoid self-medicating with antibiotics. "
        "Use medicines only according to label instructions or clinician advice."
    )
    if risk_level in {"Urgent", "High"}:
        recommendation = "Because warning signs were detected, prioritize medical evaluation instead of home-only care."

    return HealthInsight(
        possible_conditions=conditions,
        risk_level=risk_level,
        recommendation=recommendation,
        seek_medical_help=medical_help_message(risk_level),
        confidence=round(confidence, 2),
        explanation=explanation.strip(),
        sources=sources_from_docs(docs),
        metadata={"risk_triggers": risk_hits, "llm_model": model_used or "deterministic-fallback"},
    )


def answer_drug_question(question: str) -> HealthInsight:
    docs = _strict_filter_docs(question, hybrid_search(question, top_k=12), mode="drug")
    docs = [d for d in docs if _drug_doc_mentions_requested_drug(question, d)]
    if not docs:
        docs = [
            {
                "title": "Insufficient drug-specific evidence",
                "source_type": "system",
                "text": "No evidence explicitly mentioning the requested medicine was retrieved.",
                "score": 0.0,
            }
        ]
    risk_level, risk_hits = assess_risk(question)
    context = "\n\n".join(f"{doc['title']}: {doc['text']}" for doc in docs)
    llm_text, model_used = generate_text(
        "Answer this drug-information question using only the provided context. "
        "Include uses, common side effects, warnings, and when to seek help. "
        f"Question: {question}\nContext:\n{context}\nDisclaimer: {DISCLAIMER}",
        system="You are MedMind. Provide educational drug information only, not personal prescribing advice.",
        timeout=45,
    )
    explanation = llm_text or (
        "Drug information was retrieved from the local medication knowledge base. "
        "Confirm dose, interactions, pregnancy safety, allergies, and child dosing with a clinician or pharmacist."
    )
    if docs and docs[0].get("source_type") == "system":
        explanation = (
            "I could not find enough source text explicitly about the requested medicine. "
            "Try adding exact drug name, strength (e.g., 500 mg), and formulation."
        )
    return HealthInsight(
        possible_conditions=[],
        risk_level=risk_level,
        recommendation="Follow the medicine label or clinician instructions; do not combine duplicate active ingredients.",
        seek_medical_help=medical_help_message(risk_level),
        confidence=0.72 if docs else 0.35,
        explanation=explanation,
        sources=sources_from_docs(docs),
        metadata={"risk_triggers": risk_hits, "query_type": "drug_information", "llm_model": model_used or "deterministic-fallback"},
    )


def analyze_report_text(report_text: str) -> ReportInsight:
    findings: list[LabFinding] = []
    risk_blob = report_text
    extracted_labels: set[str] = set()

    for label, (pattern, low, high) in LAB_PATTERNS.items():
        match = re.search(pattern, report_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _safe_float(match.group(1))
        if value is None:
            continue
        status = "normal"
        if value < low:
            status = "low"
        elif value > high:
            status = "high"
        risk_blob += f" {label} {status}"
        extracted_labels.add(label)
        findings.append(
            LabFinding(
                test=label.title(),
                value=str(match.group(1)).replace(",", ""),
                status=status,
                explanation=f"Reference-screened as {status}; interpretation depends on age, sex, history, units, and lab range.",
            )
        )

    # OCR/table fallback for reports where value appears after multiple tokens
    # (e.g., 'GLUCOSE, FASTING, PLASMA 245.00 Very High 70-100 mg/dL').
    for label, rule in LAB_KEYWORD_RULES.items():
        if label in extracted_labels:
            continue
        token = _extract_value_by_keywords(report_text, rule["keywords"])
        if token is None:
            continue
        value = _safe_float(token)
        if value is None:
            continue
        low = float(rule["low"])
        high = float(rule["high"])
        status = "normal"
        if value < low:
            status = "low"
        elif value > high:
            status = "high"
        risk_blob += f" {label} {status}"
        findings.append(
            LabFinding(
                test=label.title(),
                value=str(token),
                status=status,
                explanation=f"OCR/table extraction matched {label}; screened as {status} against generic range {low}-{high}.",
            )
        )

    sections = _extract_sections(report_text)
    narrative_findings = _narrative_findings(report_text)
    for finding in narrative_findings:
        findings.append(finding)
        risk_blob += f" {finding.test} {finding.status}"

    docs = _strict_filter_docs(
        report_text,
        hybrid_search("medical report findings impression risk follow up " + report_text, top_k=12),
        mode="report",
    )
    if not docs:
        docs = [
            {
                "title": "Uploaded Medical Report Context",
                "source_type": "uploaded_report",
                "url": "",
                "text": report_text[:1200],
                "score": 1.0,
            }
        ]
    risk_level, _risk_hits = assess_risk(risk_blob)
    if any(finding.status in {"low", "high"} for finding in findings) and risk_level == "Low":
        risk_level = "Medium"

    abnormal_count = sum(1 for finding in findings if finding.status in {"low", "high"})
    lab_count = sum(1 for finding in findings if finding.value != "Narrative signal")
    narrative_count = len(findings) - lab_count
    llm_summary = _llm_report_summary(report_text, sections, docs)

    if findings:
        summary = (
            f"Detected {lab_count} structured lab findings and {narrative_count} narrative clinical signals. "
            f"{abnormal_count} findings are currently risk-relevant."
        )
    else:
        summary = "No clear structured findings detected from the current text. Try cleaner OCR or include more complete report sections."
    if llm_summary:
        summary = f"{summary}\n\nReport summary:\n{llm_summary}"

    recommendation = "Review the report with a clinician and correlate findings with symptoms, imaging, timeline, and treatment context."
    if risk_level in {"High", "Urgent"}:
        recommendation = "Prioritize urgent clinician review due to potential high-risk report signals."

    return ReportInsight(
        findings=findings,
        risk_level=risk_level,
        summary=summary,
        recommendation=recommendation,
        sources=sources_from_docs(docs),
    )


def to_json(data: Any) -> str:
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=2)
    return json.dumps(data, indent=2)
