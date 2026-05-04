from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DISCLAIMER = "This system is for educational purposes only and not for medical diagnosis."

OLLAMA_URL = os.getenv("MEDMIND_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BASE_URL = os.getenv("MEDMIND_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MEDMIND_OLLAMA_MODEL", "llama3.2")
OLLAMA_FALLBACK_MODEL = os.getenv("MEDMIND_OLLAMA_FALLBACK_MODEL", "llama3")
OLLAMA_VISION_MODEL = os.getenv("MEDMIND_OLLAMA_VISION_MODEL", "llava")
OLLAMA_EMBED_MODEL = os.getenv("MEDMIND_OLLAMA_EMBED_MODEL", "mxbai-embed-large")
ENABLE_OLLAMA = os.getenv("MEDMIND_ENABLE_OLLAMA", "1") != "0"
ENABLE_OLLAMA_EMBEDDINGS = os.getenv("MEDMIND_ENABLE_OLLAMA_EMBEDDINGS", "1") != "0"
ENABLE_OLLAMA_VISION = os.getenv("MEDMIND_ENABLE_OLLAMA_VISION", "1") != "0"
