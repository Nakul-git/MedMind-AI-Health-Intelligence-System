from __future__ import annotations


URGENT_PATTERNS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "confusion",
    "fainting",
    "severe dehydration",
    "blue lips",
    "unconscious",
    "seizure",
    "blood in vomit",
]

HIGH_PATTERNS = [
    "persistent vomiting",
    "bleeding",
    "stiff neck",
    "severe headache",
    "fever for 5 days",
    "pregnant",
    "immunocompromised",
]

MEDIUM_PATTERNS = [
    "fever",
    "high fever",
    "body pain",
    "cough",
    "rash",
    "diarrhea",
    "headache",
]


def assess_risk(text: str) -> tuple[str, list[str]]:
    lower = text.lower()
    urgent_hits = [pattern for pattern in URGENT_PATTERNS if pattern in lower]
    if urgent_hits:
        return "Urgent", urgent_hits
    high_hits = [pattern for pattern in HIGH_PATTERNS if pattern in lower]
    if high_hits:
        return "High", high_hits
    medium_hits = [pattern for pattern in MEDIUM_PATTERNS if pattern in lower]
    if medium_hits:
        return "Medium", medium_hits
    return "Low", []


def medical_help_message(risk_level: str) -> str:
    if risk_level == "Urgent":
        return "Seek urgent medical care now, especially if symptoms are severe, sudden, or worsening."
    if risk_level == "High":
        return "Arrange prompt clinician review today or within 24 hours, especially if symptoms worsen."
    if risk_level == "Medium":
        return "Seek medical help if symptoms persist beyond 3 days, worsen, or new warning signs appear."
    return "Monitor symptoms and consult a clinician if they persist, worsen, or concern you."

