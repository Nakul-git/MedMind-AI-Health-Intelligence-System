from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.image_analysis import analyze_medical_image
from ingestion.auto_fetch import auto_fetch_for_input
from ingestion.parse_medical_pdf import extract_pdf_text
from medmind.config import DISCLAIMER
from reasoning.diagnosis_engine import analyze_report_text, analyze_symptoms, answer_drug_question
from retrieval.hybrid_search import hybrid_search


st.set_page_config(page_title="MedMind", page_icon="M", layout="wide")

st.title("MedMind")
st.caption("Multimodal Clinical RAG + Structured Decision Support")
st.warning(DISCLAIMER)

mode = st.sidebar.radio(
    "Mode",
    ["Symptom Assistant", "Medical Report Analyzer", "X-ray / Image Analyzer", "Drug Information RAG", "Knowledge Retrieval"],
)


def show_json(payload) -> None:
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload
    st.json(data)


if mode == "Symptom Assistant":
    st.subheader("Symptom-to-Insight Assistant")
    auto_fetch = st.checkbox("Fetch fresh PubMed / open-access PDF evidence before analysis", value=False)
    symptoms = st.text_area(
        "Symptoms",
        value="I have fever, headache, and body pain for 3 days",
        height=130,
    )
    if st.button("Analyze Symptoms", type="primary"):
        if auto_fetch:
            with st.spinner("Fetching PubMed summaries and open-access PDFs, then rebuilding the RAG index..."):
                fetch_result = auto_fetch_for_input(symptoms, mode="symptoms", download_pdfs=True)
            st.success(f"Fetched {fetch_result['documents_found']} docs, added {fetch_result['documents_added']} new docs.")
            if fetch_result["pdfs_downloaded"]:
                st.caption(f"PDFs downloaded: {len(fetch_result['pdfs_downloaded'])}")
        result = analyze_symptoms(symptoms)
        left, right = st.columns([1, 1])
        with left:
            st.metric("Risk Level", result.risk_level)
            st.write(result.recommendation)
            st.write(result.seek_medical_help)
            st.subheader("Possible Conditions")
            for condition in result.possible_conditions:
                st.write(f"**{condition.name}** - {condition.likelihood}")
                st.caption(", ".join(condition.evidence))
        with right:
            st.subheader("Structured JSON")
            show_json(result)
        st.subheader("Sources")
        for source in result.sources:
            st.write(f"**{source.title}** - {source.source_type} - score {source.score:.3f}")
            st.caption(source.snippet)

elif mode == "Medical Report Analyzer":
    st.subheader("Medical Report Analyzer")
    auto_fetch = st.checkbox("Fetch fresh lab/report evidence before analysis", value=False)
    uploaded = st.file_uploader("Upload PDF, image, or text report", type=["pdf", "png", "jpg", "jpeg", "webp", "txt"])
    report_text = st.text_area("Or paste report text", height=180)

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        temp = ROOT / "data" / "reports" / f"_uploaded{suffix}"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(uploaded.getvalue())
        if suffix == ".pdf":
            report_text = extract_pdf_text(temp)
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            report_text = analyze_medical_image(temp).extracted_text
        else:
            report_text = temp.read_text(encoding="utf-8", errors="ignore")
        temp.unlink(missing_ok=True)
        st.text_area("Extracted text", value=report_text, height=180)

    if st.button("Analyze Report", type="primary"):
        if auto_fetch:
            with st.spinner("Fetching report/lab evidence and rebuilding the RAG index..."):
                fetch_result = auto_fetch_for_input(report_text, mode="report", download_pdfs=True)
            st.success(f"Fetched {fetch_result['documents_found']} docs, added {fetch_result['documents_added']} new docs.")
        result = analyze_report_text(report_text)
        st.metric("Risk Level", result.risk_level)
        st.write(result.summary)
        st.write(result.recommendation)
        st.subheader("Findings")
        for finding in result.findings:
            st.write(f"**{finding.test}**: {finding.value} - {finding.status}")
            st.caption(finding.explanation)
        st.subheader("Structured JSON")
        show_json(result)

elif mode == "X-ray / Image Analyzer":
    st.subheader("X-ray / Image Analyzer")
    st.caption("Uses LLaVA through Ollama when available. X-ray output is not a radiology diagnosis.")
    auto_fetch = st.checkbox("Fetch radiology PubMed / open-access PDF evidence after image analysis", value=True)
    uploaded = st.file_uploader("Upload X-ray, scan, prescription, or report image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        temp = ROOT / "data" / "reports" / f"_image_upload{suffix}"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(uploaded.getvalue())
        st.image(str(temp), use_container_width=True)
        if st.button("Analyze Image", type="primary"):
            result = analyze_medical_image(temp)
            if auto_fetch:
                xray_query = f"{result.image_type} {result.extracted_text} {result.visual_summary[:700]}"
                with st.spinner("Fetching radiology evidence and rebuilding the RAG index..."):
                    fetch_result = auto_fetch_for_input(xray_query, mode="xray", download_pdfs=True)
                st.success(f"Fetched {fetch_result['documents_found']} docs, added {fetch_result['documents_added']} new docs.")
                if fetch_result["pdfs_downloaded"]:
                    st.caption(f"PDFs downloaded: {len(fetch_result['pdfs_downloaded'])}")
            st.metric("Image Type", result.image_type)
            st.write(result.recommendation)
            st.subheader("Visual Summary")
            st.write(result.visual_summary)
            if result.extracted_text:
                st.subheader("Extracted Text")
                st.text(result.extracted_text)
            st.subheader("Limitations")
            for item in result.limitations:
                st.caption(item)
            st.subheader("Structured JSON")
            show_json(result)
        temp.unlink(missing_ok=True)

elif mode == "Drug Information RAG":
    st.subheader("Drug Information RAG")
    auto_fetch = st.checkbox("Fetch fresh drug evidence before search", value=False)
    question = st.text_input("Question", value="What are Paracetamol side effects and warnings?")
    if st.button("Search Drug Knowledge", type="primary"):
        if auto_fetch:
            with st.spinner("Fetching drug evidence and rebuilding the RAG index..."):
                fetch_result = auto_fetch_for_input(question, mode="drug", download_pdfs=True)
            st.success(f"Fetched {fetch_result['documents_found']} docs, added {fetch_result['documents_added']} new docs.")
        result = answer_drug_question(question)
        st.write(result.explanation)
        st.write(result.recommendation)
        for source in result.sources:
            st.write(f"**{source.title}**")
            st.caption(source.snippet)
        st.subheader("Structured JSON")
        show_json(result)

else:
    st.subheader("Hybrid Clinical Retrieval")
    query = st.text_input("Search query", value="fever headache body pain warning signs")
    if st.button("Retrieve", type="primary"):
        results = hybrid_search(query, top_k=8)
        for result in results:
            st.write(f"**{result['title']}** - {result['source_type']} - {result['score']:.3f}")
            st.caption(result["text"][:350])
        with st.expander("Raw results"):
            st.code(json.dumps(results, indent=2), language="json")
