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

```json
{
  "possible_conditions": [
    {
      "name": "Viral fever",
      "likelihood": "medium",
      "evidence": ["fever", "headache", "body pain"]
    }
  ],
  "risk_level": "Medium",
  "recommendation": "Rest, hydrate, monitor temperature, and seek care if symptoms worsen.",
  "seek_medical_help": "Seek urgent care for chest pain, breathing difficulty, confusion, severe dehydration, or fever persisting beyond 3 days.",
  "confidence": 0.67,
  "disclaimer": "This system is for educational purposes only and not for medical diagnosis."
}
```
