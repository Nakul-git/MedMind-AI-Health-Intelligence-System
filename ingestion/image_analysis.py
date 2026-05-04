from __future__ import annotations

from pathlib import Path

from ingestion.ocr_reports import extract_image_text
from medmind.config import DISCLAIMER
from medmind.models import ImageInsight
from medmind.ollama_client import generate_vision


def _classify_image_type(path: Path, ocr_text: str) -> str:
    lower = f"{path.name} {ocr_text}".lower()
    if any(term in lower for term in ["xray", "x-ray", "radiograph", "chest pa", "ap view"]):
        return "xray"
    if any(term in lower for term in ["prescription", "rx", "tablet", "capsule"]):
        return "prescription"
    if any(term in lower for term in ["hemoglobin", "platelets", "glucose", "wbc", "cbc"]):
        return "lab_report_image"
    return "medical_image"


def analyze_medical_image(path: str | Path) -> ImageInsight:
    """Analyze a medical image with OCR plus an optional Ollama vision model.

    This is decision support, not radiology diagnosis. X-ray findings must be
    reviewed by a licensed clinician or radiologist.
    """
    image_path = Path(path)
    text = extract_image_text(path)
    image_type = _classify_image_type(image_path, text)
    prompt = (
        "You are MedMind, a cautious medical image education assistant. "
        "Inspect this image and return a concise, structured plain-text analysis with these labels: "
        "Image type, Visible observations, Possible findings, Urgency cues, Limitations, Next step. "
        "If this is an X-ray, describe visible radiographic patterns only and say it requires radiologist review. "
        "Do not provide a definitive diagnosis. Do not state that an image is normal, clear, safe, or free of urgent findings. "
        "Say that urgency cannot be determined from the image alone and depends on symptoms and clinician review. "
        f"OCR text if any: {text or 'none'}\nDisclaimer: {DISCLAIMER}"
    )
    vision_text, model_used = generate_vision(prompt, image_path)

    limitations = [
        "Educational support only; not a diagnostic radiology read.",
        "Image quality, projection, missing clinical history, and model limitations can affect observations.",
    ]
    if not vision_text:
        limitations.append("Vision model was unavailable; only OCR text was extracted.")

    possible_findings = []
    if vision_text:
        for line in vision_text.splitlines():
            cleaned = line.strip(" -*")
            if cleaned and any(term in cleaned.lower() for term in ["finding", "opacity", "fracture", "consolidation", "effusion", "normal", "abnormal"]):
                possible_findings.append(cleaned[:220])
    if not possible_findings and text:
        possible_findings.append("Text-bearing medical image; route extracted text through report or drug analysis.")

    safety_note = (
        "\n\nSafety note: MedMind cannot rule out disease, confirm a normal X-ray, or determine urgency from an image alone."
    )

    recommendation = (
        "For X-rays or scans, get a clinician/radiologist review, especially for chest pain, breathing difficulty, trauma, fever, or worsening symptoms."
        if image_type == "xray"
        else "Use this as an extraction and triage aid; confirm interpretation with a qualified clinician."
    )

    return ImageInsight(
        modality="image_or_scan",
        image_type=image_type,
        extracted_text=text,
        visual_summary=(vision_text + safety_note) if vision_text else "No multimodal visual summary available.",
        possible_findings=possible_findings[:6],
        limitations=limitations,
        recommendation=recommendation,
        model_used=model_used or "ocr-fallback",
    )
