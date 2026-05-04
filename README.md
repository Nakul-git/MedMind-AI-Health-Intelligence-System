<<<<<<< HEAD
# 🏥 MedMind - AI Health Intelligence System

> A **multimodal AI health intelligence system** combining Clinical RAG, medical document understanding, symptom reasoning, drug information retrieval, report analysis, and X-ray/image interpretation into one educational decision-support platform.

> ⚠️ **Disclaimer:** MedMind is for **educational purposes only** and is not intended for medical diagnosis. Always consult a qualified clinician for medical decisions or urgent symptoms.

---

## 📌 What is MedMind?

MedMind is not just a medical chatbot.

It takes **symptoms, lab reports, PDFs, prescriptions, and medical images** retrieves relevant clinical evidence, reasons over the information, and returns **structured insights** with risk levels, possible conditions, recommendations, and source-backed explanations.

---

## ⚡ System Architecture — Full Flow

```
User Input (symptoms / report / image / drug query)
       │
       ▼
┌──────────────────────────┐
│     Ingestion Layer      │  fetch → parse → OCR → embed → store
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Auto-Fetch Evidence     │  PubMed + Europe PMC → PDF → knowledge base
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│    Retrieval Layer       │  multi-query → lexical + semantic → RRF → rerank
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│    Reasoning Layer       │  diagnosis engine → risk engine → structured output
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│     API / UI Layer       │  FastAPI (REST) + Streamlit (UI)
└──────────────────────────┘
```

---

## 🏗️ Repository Structure

```
medmind/
├── api/
│   └── app.py                      # FastAPI REST interface
│
├── data/
│   ├── drug_data/                  # Local drug information knowledge base
│   ├── embeddings/                 # Stored vector embeddings
│   ├── knowledge_base/             # Clinical evidence + auto-ingested data
│   ├── medical_papers/             # Downloaded PubMed / PMC PDFs
│   └── reports/                    # Uploaded patient reports and scans
│
├── ingestion/
│   ├── auto_fetch.py               # Orchestrates auto evidence fetching
│   ├── embed_store.py              # Generates embeddings → stores in vector DB
│   ├── fetch_pubmed.py             # Fetches summaries from PubMed / Europe PMC
│   ├── image_analysis.py           # LLaVA multimodal image understanding
│   ├── ocr_reports.py              # OCR extraction from scans and prescriptions
│   └── parse_medical_pdf.py        # Parses and structures medical PDFs
│
├── medmind/
│   ├── config.py                   # Centralized configuration management
│   ├── models.py                   # Pydantic data models and schemas
│   └── ollama_client.py            # Ollama model client and routing
│
├── reasoning/
│   ├── diagnosis_engine.py         # Symptom-to-condition reasoning
│   ├── risk_engine.py              # Red-flag detection and risk scoring
│   └── structured_output.py       # Formats output as structured JSON
│
├── retrieval/
│   ├── hybrid_search.py            # Lexical + semantic hybrid search
│   ├── medical_reranker.py         # Medical-domain-aware reranking
│   ├── multi_query.py              # Expands query into multiple search angles
│   └── rrf.py                      # Reciprocal Rank Fusion scoring
│
├── ui/
│   └── streamlit_app.py            # Streamlit web UI
│
├── main.py                         # Application entry point
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🔍 Layer Breakdown

### 📥 Ingestion Layer

Handles everything from raw uploads to searchable clinical knowledge.

```
ingestion/
├── fetch_pubmed.py        # Queries PubMed + Europe PMC for clinical evidence
├── parse_medical_pdf.py   # Extracts and structures text from medical PDFs
├── ocr_reports.py         # OCR on scans, prescriptions, and lab reports
├── image_analysis.py      # LLaVA visual analysis for X-rays and images
├── embed_store.py         # Embeds content with mxbai-embed-large → vector DB
└── auto_fetch.py          # Full auto-fetch pipeline orchestration
```

**Pipeline:**
```
Input → parse / OCR → image_analysis → embed_store → knowledge base
```

---

### 🔍 Retrieval Layer

Multi-stage hybrid retrieval with medical-aware reranking.

```
retrieval/
├── multi_query.py         # Expands a single query into multiple search angles
├── hybrid_search.py       # Lexical retrieval + mxbai-embed-large semantic search
├── rrf.py                 # Reciprocal Rank Fusion to merge ranked results
└── medical_reranker.py    # Medical-domain reranking for clinical precision
```

**Pipeline:**
```
Query → multi_query → lexical search + semantic search → RRF fusion → medical_reranker → context
```

---

### 🧠 Reasoning Layer

Clinical reasoning, risk detection, and structured output generation.

```
reasoning/
├── diagnosis_engine.py    # Maps symptoms + evidence to possible conditions
├── risk_engine.py         # Detects red-flag symptoms and assigns risk levels
└── structured_output.py   # Returns validated, structured JSON insights
```

**Pipeline:**
```
Retrieved context → diagnosis_engine → risk_engine → structured_output → JSON response
```

---

### 🤖 AI / Model Stack

MedMind runs **fully locally** via Ollama — no external API required.

| Model | Role |
|-------|------|
| `mxbai-embed-large` | Semantic embeddings for retrieval |
| `llava` | Medical image and X-ray visual analysis |
| `llama3.2-vision` | Vision fallback for image understanding |
| `llama3.2` | Primary clinical reasoning |
| `llama3` | Text reasoning fallback |

> If Ollama is unavailable, MedMind falls back to deterministic retrieval and rule-based safety logic.

---

## 🔄 Retrieval Pipeline

```
User Input
   │
   ▼
Multi-query expansion
   │
   ▼
Lexical retrieval
   │
   ▼
mxbai-embed-large semantic retrieval
   │
   ▼
RRF fusion
   │
   ▼
Medical reranking
   │
   ▼
Retrieved clinical context
   │
   ▼
Reasoning layer
   │
   ▼
Structured educational output
```

---

## 🌐 Auto-Fetch Evidence Pipeline

MedMind can fetch live clinical evidence automatically — no manual data entry required.

```
User query / report / X-ray summary
   │
   ▼
Build medical search query
   │
   ▼
Fetch PubMed summaries
   │
   ▼
Search Europe PMC open-access articles
   │
   ▼
Download relevant PDFs
   │
   ▼
Extract PDF text
   │
   ▼
Save into local knowledge base
   │
   ▼
Rebuild vector index
   │
   ▼
Answer with updated clinical evidence
```

**Fetched data is stored in:**
```
data/knowledge_base/auto_ingested.jsonl
data/medical_papers/pdfs/
```

---

## 🩻 X-ray / Image Workflow

```
Uploaded image
   │
   ▼
OCR extraction
   │
   ▼
Image type classification
   │
   ▼
LLaVA visual analysis
   │
   ▼
llama3.2-vision fallback if needed
   │
   ▼
Radiology-focused evidence fetch
   │
   ▼
Structured output with limitations and recommendations
```

> MedMind does **not** provide radiology diagnosis. It provides educational observations and always recommends clinician/radiologist review.

---

## 📤 Example Output
=======
# MedMind

MedMind is an AI health intelligence system prototype for multimodal clinical RAG and decision support.

> This system is for educational purposes only and not for medical diagnosis. Always consult a qualified clinician for medical decisions or urgent symptoms.

## What It Does

- Symptom-to-insight assistant with ranked possible conditions
- Medical report analyzer for pasted lab values or uploaded text/PDF content
- Drug information RAG over a local medication knowledge base
- X-ray, scan, prescription, and report-image workflow using LLaVA/Ollama plus OCR fallback
- Hybrid retrieval with multi-query expansion, BM25-style lexical search, vector similarity, RRF fusion, and optional reranking
- Structured JSON output for downstream apps

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py ingest
uvicorn api.app:app --reload
```

In another terminal:

```powershell
streamlit run ui/streamlit_app.py
```

## Auto-Fetch Evidence

MedMind can fetch evidence for an input instead of requiring you to manually place every file in `data/`.

From the UI, enable the fetch checkbox in:

- Symptom Assistant
- Medical Report Analyzer
- X-ray / Image Analyzer
- Drug Information RAG

From the CLI:

```powershell
python main.py auto-fetch --query "fever headache body pain for 3 days" --mode symptoms
python main.py auto-fetch --query "chest xray pneumonia opacity" --mode xray
python main.py auto-fetch --query "paracetamol side effects warnings" --mode drug
```

The auto-fetch pipeline:

1. Builds a medical search query from the input.
2. Fetches PubMed summaries through NCBI E-utilities.
3. Searches Europe PMC for open-access articles.
4. Downloads open-access PDFs when available.
5. Extracts PDF text.
6. Writes reusable records into `data/knowledge_base/auto_ingested.jsonl`.
7. Rebuilds the RAG index with `mxbai-embed-large` embeddings.

Downloaded PDFs are stored in:

```text
data/medical_papers/pdfs/
```

## Ollama Models

MedMind is configured for the local Ollama stack below:

- Embeddings: `mxbai-embed-large`
- Vision / X-ray analysis: `llava`, with `llama3.2-vision` as a secondary fallback
- Text reasoning: `llama3.2`, with `llama3` fallback

Pull the models if needed:

```powershell
ollama pull mxbai-embed-large
ollama pull llava
ollama pull llama3.2
ollama pull llama3
```

Optional overrides:

```powershell
$env:MEDMIND_OLLAMA_MODEL="llama3.2"
$env:MEDMIND_OLLAMA_FALLBACK_MODEL="llama3"
$env:MEDMIND_OLLAMA_VISION_MODEL="llava"
$env:MEDMIND_OLLAMA_EMBED_MODEL="mxbai-embed-large"
```

Set `MEDMIND_ENABLE_OLLAMA=0` to force deterministic fallback mode.

## Project Layout

```text
medmind/
├── data/
│   ├── medical_papers/
│   ├── drug_data/
│   ├── reports/
│   ├── knowledge_base/
│   └── embeddings/
├── ingestion/
├── retrieval/
├── reasoning/
├── api/
├── ui/
└── main.py
```

## Example Structured Output
>>>>>>> 8462347 (Initial MedMind AI health intelligence system)

```json
{
  "possible_conditions": [
    {
      "name": "Viral fever",
<<<<<<< HEAD
      "likelihood": "high",
=======
      "likelihood": "medium",
>>>>>>> 8462347 (Initial MedMind AI health intelligence system)
      "evidence": ["fever", "headache", "body pain"]
    }
  ],
  "risk_level": "Medium",
<<<<<<< HEAD
  "recommendation": "Rest, hydrate, monitor symptoms, and seek care if symptoms worsen.",
  "seek_medical_help": "Seek medical help if symptoms persist beyond 3 days or warning signs appear.",
  "confidence": 0.86,
  "disclaimer": "This system is for educational purposes only and not for medical diagnosis."
}
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/models` | List loaded Ollama models |
| `POST` | `/symptoms` | Symptom analysis and condition reasoning |
| `POST` | `/drug` | Drug information retrieval |
| `POST` | `/report` | Lab report / blood test analysis |
| `POST` | `/retrieve` | Raw clinical evidence retrieval |
| `POST` | `/image` | X-ray / scan / prescription image analysis |
| `POST` | `/upload-report` | Upload a report file for analysis |
| `POST` | `/auto-fetch` | Trigger auto-fetch from PubMed / Europe PMC |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Embeddings | mxbai-embed-large (Ollama) |
| Image Reasoning | LLaVA + llama3.2-vision |
| Clinical Reasoning | llama3.2 / llama3 (Ollama) |
| Hybrid Search | Lexical + Semantic + RRF |
| Reranking | Medical-aware reranker |
| Evidence Sources | PubMed / NCBI E-utilities + Europe PMC |
| PDF Parsing | Custom medical PDF parser |
| OCR | Report and prescription OCR |
| API | FastAPI |
| UI | Streamlit |

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Ollama models
```bash
ollama pull mxbai-embed-large
ollama pull llava
ollama pull llama3.2
ollama pull llama3
ollama pull llama3.2-vision
```

### 3. Run ingestion
```bash
python main.py ingest
```

### 4. Start the API
```bash
uvicorn api.app:app --reload
```

### 5. Launch the UI
```bash
streamlit run ui/streamlit_app.py
```

Open: **http://127.0.0.1:8501**

---

## 🔁 Auto-Fetch Examples

```bash
# Symptom-based evidence fetch
python main.py auto-fetch --query "fever headache body pain for 3 days" --mode symptoms

# X-ray evidence fetch
python main.py auto-fetch --query "chest xray pneumonia opacity" --mode xray

# Drug information fetch
python main.py auto-fetch --query "paracetamol side effects warnings" --mode drug
```

---

## 🛡️ Safety

MedMind is designed as an **educational clinical AI prototype**.

It does **not** replace:
- Doctors or clinicians
- Radiologists
- Pharmacists
- Emergency care
- Professional medical judgment

> 🚨 **Urgent symptoms** - chest pain, breathing difficulty, confusion, fainting, severe dehydration, seizures, or rapidly worsening illness - require **immediate medical care**.

---

> ⚠️ **MedMind is for educational purposes only. It is not a diagnostic tool. Always consult a qualified medical professional.**
=======
  "recommendation": "Rest, hydrate, monitor temperature, and seek care if symptoms worsen.",
  "seek_medical_help": "Seek urgent care for chest pain, breathing difficulty, confusion, severe dehydration, or fever persisting beyond 3 days.",
  "confidence": 0.67,
  "disclaimer": "This system is for educational purposes only and not for medical diagnosis."
}
```
>>>>>>> 8462347 (Initial MedMind AI health intelligence system)
