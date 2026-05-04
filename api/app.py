from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from ingestion.auto_fetch import auto_fetch_for_input
from ingestion.image_analysis import analyze_medical_image
from ingestion.parse_medical_pdf import extract_pdf_text
from medmind.config import DISCLAIMER
from medmind.models import HealthInsight, ImageInsight, ReportInsight
from medmind.ollama_client import model_status
from reasoning.diagnosis_engine import analyze_report_text, analyze_symptoms, answer_drug_question
from retrieval.hybrid_search import hybrid_search


app = FastAPI(
    title="MedMind API",
    description="Educational AI health intelligence system with clinical RAG and structured decision support.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str


class ReportRequest(BaseModel):
    report_text: str


class AutoFetchRequest(BaseModel):
    query: str
    mode: str = "symptoms"
    download_pdfs: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/models")
def models() -> dict:
    return {**model_status(), "disclaimer": DISCLAIMER}


@app.post("/symptoms", response_model=HealthInsight)
def symptoms(request: QueryRequest) -> HealthInsight:
    return analyze_symptoms(request.query)


@app.post("/drug", response_model=HealthInsight)
def drug(request: QueryRequest) -> HealthInsight:
    return answer_drug_question(request.query)


@app.post("/report", response_model=ReportInsight)
def report(request: ReportRequest) -> ReportInsight:
    return analyze_report_text(request.report_text)


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> dict:
    return {"results": hybrid_search(request.query, top_k=8), "disclaimer": DISCLAIMER}


@app.post("/auto-fetch")
def auto_fetch(request: AutoFetchRequest) -> dict:
    result = auto_fetch_for_input(request.query, mode=request.mode, download_pdfs=request.download_pdfs)
    return {**result, "disclaimer": DISCLAIMER}


@app.post("/image", response_model=ImageInsight)
async def image(file: UploadFile = File(...)) -> ImageInsight:
    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await file.read())
        temp_path = Path(handle.name)
    result = analyze_medical_image(temp_path)
    temp_path.unlink(missing_ok=True)
    return result


@app.post("/upload-report", response_model=ReportInsight)
async def upload_report(file: UploadFile = File(...)) -> ReportInsight:
    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await file.read())
        temp_path = Path(handle.name)

    if suffix == ".pdf":
        text = extract_pdf_text(temp_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        text = analyze_medical_image(temp_path).extracted_text
    else:
        text = temp_path.read_text(encoding="utf-8", errors="ignore")
    temp_path.unlink(missing_ok=True)
    return analyze_report_text(text)
