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

st.markdown(
    """
    <style>
    .mm-hero-wrap {
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
        padding: 1.0rem 1.1rem 0.9rem 1.1rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(30,58,138,0.35) 0%, rgba(2,6,23,0.10) 40%, rgba(15,23,42,0.28) 100%);
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 12px 35px rgba(2,6,23,0.25);
    }
    .mm-hero-title {
        margin: 0;
        font-size: clamp(2.1rem, 3vw, 3rem);
        font-weight: 850;
        letter-spacing: 0.4px;
        line-height: 1.05;
        background: linear-gradient(92deg, #f8fafc 0%, #93c5fd 35%, #38bdf8 65%, #22d3ee 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 0 20px rgba(56,189,248,0.18);
    }
    .mm-hero-sub {
        margin-top: 0.4rem;
        margin-bottom: 0;
        color: #cbd5e1;
        font-size: 1.02rem;
        letter-spacing: 0.25px;
    }
    .mm-section-title {
        margin-top: 0.7rem;
        margin-bottom: 0.55rem;
        font-size: clamp(1.45rem, 1.9vw, 2rem);
        font-weight: 780;
        line-height: 1.2;
        color: #e2e8f0;
        letter-spacing: 0.2px;
        border-left: 4px solid #38bdf8;
        padding-left: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mm-hero-wrap">
        <h1 class="mm-hero-title">MedMind</h1>
        <p class="mm-hero-sub">Multimodal Clinical RAG + Structured Decision Support</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(DISCLAIMER)

mode = st.sidebar.radio(
    "Mode",
    ["Symptom Assistant", "Medical Report Analyzer", "X-ray / Image Analyzer", "Drug Information RAG", "Knowledge Retrieval"],
)


def show_json(payload) -> None:
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload
    st.json(data)


if mode == "Symptom Assistant":
    st.markdown('<div class="mm-section-title">Symptom-to-Insight Assistant</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="mm-section-title">Possible Conditions</div>', unsafe_allow_html=True)
            for condition in result.possible_conditions:
                st.write(f"**{condition.name}** - {condition.likelihood}")
                st.caption(", ".join(condition.evidence))
        with right:
            st.markdown('<div class="mm-section-title">Structured JSON</div>', unsafe_allow_html=True)
            show_json(result)
        st.markdown('<div class="mm-section-title">Sources</div>', unsafe_allow_html=True)
        for source in result.sources:
            st.write(f"**{source.title}** - {source.source_type} - score {source.score:.3f}")
            st.caption(source.snippet)

elif mode == "Medical Report Analyzer":
    st.markdown('<div class="mm-section-title">Medical Report Analyzer</div>', unsafe_allow_html=True)
    st.caption("Clinical-grade workflow: upload -> verify extracted text -> analyze findings with risk and evidence.")
    auto_fetch = st.checkbox("Fetch fresh lab/report evidence before analysis", value=False)

    upload_col, qa_col = st.columns([1.3, 1.0])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload report (PDF, image, or text)",
            type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
            help="Best results: machine-readable PDF or clear lab image.",
        )
    with qa_col:
        st.info("Tip: confirm extracted text before running analysis to reduce missed values.")

    extracted_text = ""
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        temp = ROOT / "data" / "reports" / f"_uploaded{suffix}"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(uploaded.getvalue())
        if suffix == ".pdf":
            extracted_text = extract_pdf_text(temp)
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            extracted_text = analyze_medical_image(temp).extracted_text
        else:
            extracted_text = temp.read_text(encoding="utf-8", errors="ignore")
        temp.unlink(missing_ok=True)

    report_text = st.text_area(
        "Report Text (edit/clean before analysis)",
        value=extracted_text,
        height=180,
        placeholder="Paste lab/report text here if upload extraction is noisy...",
    )

    if extracted_text:
        with st.expander("Extraction Quality Check", expanded=False):
            st.write("Preview from file extraction:")
            st.code(extracted_text[:5000], language="text")

    if st.button("Analyze Report", type="primary", use_container_width=True):
        if auto_fetch:
            with st.spinner("Fetching report/lab evidence and rebuilding the RAG index..."):
                fetch_result = auto_fetch_for_input(
                    report_text,
                    mode="report",
                    download_pdfs=True,
                    local_report_text=report_text,
                    local_report_name=(uploaded.name if uploaded else "Pasted Medical Report"),
                )
            st.success(f"Fetched {fetch_result['documents_found']} docs, added {fetch_result['documents_added']} new docs.")
            if fetch_result.get("pdfs_downloaded"):
                st.caption(f"Downloaded PDFs: {len(fetch_result['pdfs_downloaded'])}")
            if fetch_result.get("query"):
                st.caption(f"Fetch query: {fetch_result['query'][:300]}")
        result = analyze_report_text(report_text)

        risk_col, summary_col = st.columns([0.35, 0.65])
        with risk_col:
            st.metric("Risk Level", result.risk_level)
        with summary_col:
            st.write(f"**Summary:** {result.summary}")
            st.write(f"**Recommendation:** {result.recommendation}")

        st.markdown('<div class="mm-section-title">Findings</div>', unsafe_allow_html=True)
        if result.findings:
            finding_rows = [
                {
                    "Test": f.test,
                    "Value": f.value,
                    "Status": f.status.upper(),
                    "Interpretation": f.explanation,
                }
                for f in result.findings
            ]
            st.dataframe(finding_rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No structured findings detected from this text. Try cleaner OCR or include clearer report sections (Findings/Impression/Plan).")

        if result.sources:
            st.markdown('<div class="mm-section-title">Evidence Sources</div>', unsafe_allow_html=True)
            for source in result.sources:
                st.write(f"**{source.title}** - score {source.score:.3f}")
                st.caption(source.snippet)

        st.markdown('<div class="mm-section-title">Structured JSON</div>', unsafe_allow_html=True)
        show_json(result)

elif mode == "X-ray / Image Analyzer":
    st.markdown('<div class="mm-section-title">X-ray / Image Analyzer</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="mm-section-title">Visual Summary</div>', unsafe_allow_html=True)
            st.write(result.visual_summary)
            if result.extracted_text:
                st.markdown('<div class="mm-section-title">Extracted Text</div>', unsafe_allow_html=True)
                st.text(result.extracted_text)
            st.markdown('<div class="mm-section-title">Limitations</div>', unsafe_allow_html=True)
            for item in result.limitations:
                st.caption(item)
            st.markdown('<div class="mm-section-title">Structured JSON</div>', unsafe_allow_html=True)
            show_json(result)
        temp.unlink(missing_ok=True)

elif mode == "Drug Information RAG":
    st.markdown('<div class="mm-section-title">Drug Information RAG</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="mm-section-title">Structured JSON</div>', unsafe_allow_html=True)
        show_json(result)

else:
    st.markdown('<div class="mm-section-title">Hybrid Clinical Retrieval</div>', unsafe_allow_html=True)
    query = st.text_input("Search query", value="fever headache body pain warning signs")
    if st.button("Retrieve", type="primary"):
        results = hybrid_search(query, top_k=8)
        for result in results:
            st.write(f"**{result['title']}** - {result['source_type']} - {result['score']:.3f}")
            st.caption(result["text"][:350])
        with st.expander("Raw results"):
            st.code(json.dumps(results, indent=2), language="json")
