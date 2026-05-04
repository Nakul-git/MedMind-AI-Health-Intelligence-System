from __future__ import annotations


MEDICAL_SYNONYMS = {
    "fever": ["high temperature", "pyrexia"],
    "headache": ["head pain", "migraine symptom"],
    "body pain": ["myalgia", "muscle aches"],
    "cough": ["dry cough", "respiratory symptom"],
    "chest pain": ["cardiac warning sign", "acute chest discomfort"],
    "rash": ["skin eruption", "exanthem"],
    "paracetamol": ["acetaminophen", "fever medicine"],
}


def expand_query(query: str) -> list[str]:
    expanded = [query]
    lower = query.lower()
    for phrase, synonyms in MEDICAL_SYNONYMS.items():
        if phrase in lower:
            expanded.extend(f"{query} {synonym}" for synonym in synonyms)
    return list(dict.fromkeys(expanded))[:6]

