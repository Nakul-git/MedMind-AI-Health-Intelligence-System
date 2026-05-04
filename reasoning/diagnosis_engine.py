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

LAB_PATTERNS = {
    "hemoglobin": (r"(?:hemoglobin|hb)\s*[:\-]?\s*([0-9.]+)\s*(g/dl)?", 12.0, 16.0),
    "white blood cells": (r"(?:white blood cells|wbc)\s*[:\-]?\s*([0-9.]+)", 4000.0, 11000.0),
    "platelets": (r"(?:platelets|platelet count)\s*[:\-]?\s*([0-9.]+)", 150000.0, 450000.0),
    "fasting glucose": (r"(?:fasting glucose|glucose)\s*[:\-]?\s*([0-9.]+)", 70.0, 99.0),
    "hba1c": (r"(?:hba1c|a1c)\s*[:\-]?\s*([0-9.]+)", 4.0, 5.6),
}


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
    docs = hybrid_search(symptoms, top_k=5)
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
    docs = hybrid_search(question, top_k=5)
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
    for label, (pattern, low, high) in LAB_PATTERNS.items():
        match = re.search(pattern, report_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        status = "normal"
        if value < low:
            status = "low"
        elif value > high:
            status = "high"
        risk_blob += f" {label} {status}"
        findings.append(
            LabFinding(
                test=label.title(),
                value=match.group(1),
                status=status,
                explanation=f"Reference-screened as {status}; interpretation depends on age, sex, history, units, and lab range.",
            )
        )

    docs = hybrid_search("lab report CBC glucose abnormal values " + report_text, top_k=5)
    risk_level, _risk_hits = assess_risk(risk_blob)
    if any(finding.status in {"low", "high"} for finding in findings) and risk_level == "Low":
        risk_level = "Medium"

    abnormal_count = sum(1 for finding in findings if finding.status != "normal")
    summary = (
        f"Detected {len(findings)} lab values; {abnormal_count} appear outside generic screening ranges."
        if findings
        else "No supported lab values were detected. Paste clearer text or upload a text-readable report."
    )

    return ReportInsight(
        findings=findings,
        risk_level=risk_level,
        summary=summary,
        recommendation="Review abnormal results with a clinician, especially if symptoms are present or values differ from the lab's own reference range.",
        sources=sources_from_docs(docs),
    )


def to_json(data: Any) -> str:
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=2)
    return json.dumps(data, indent=2)
