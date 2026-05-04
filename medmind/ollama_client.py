from __future__ import annotations

import base64
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from medmind.config import (
    ENABLE_OLLAMA,
    ENABLE_OLLAMA_EMBEDDINGS,
    ENABLE_OLLAMA_VISION,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_MODEL,
    OLLAMA_VISION_MODEL,
)


def _post_json(path: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _get_json(path: str, timeout: int = 3) -> dict[str, Any] | None:
    request = urllib.request.Request(f"{OLLAMA_BASE_URL}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def model_status(timeout: int = 8) -> dict[str, Any]:
    installed: list[str] = []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        lines = result.stdout.splitlines()[1:]
        installed = sorted(line.split()[0] for line in lines if line.strip())
    except (OSError, subprocess.TimeoutExpired):
        installed = []

    def has_model(name: str) -> bool:
        return any(item == name or item.startswith(f"{name}:") for item in installed)

    configured = {
        "text_primary": OLLAMA_MODEL,
        "text_fallback": OLLAMA_FALLBACK_MODEL,
        "vision_primary": OLLAMA_VISION_MODEL,
        "vision_fallback": "llama3.2-vision",
        "embedding": OLLAMA_EMBED_MODEL,
    }
    availability = {role: has_model(model) for role, model in configured.items()}
    return {
        "ollama_available": bool(installed),
        "configured": configured,
        "availability": availability,
        "installed_models": installed,
    }


def generate_text(prompt: str, system: str | None = None, timeout: int = 60) -> tuple[str | None, str | None]:
    if not ENABLE_OLLAMA:
        return None, None

    for model in [OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if system:
            payload["system"] = system
        data = _post_json("/api/generate", payload, timeout=timeout)
        text = (data or {}).get("response")
        if text:
            return text, model
    return None, None


def generate_vision(prompt: str, image_path: str | Path, timeout: int = 120) -> tuple[str | None, str | None]:
    if not ENABLE_OLLAMA_VISION:
        return None, None

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    for model in [OLLAMA_VISION_MODEL, "llama3.2-vision"]:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [encoded],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        data = _post_json("/api/generate", payload, timeout=timeout)
        text = (data or {}).get("response")
        if text:
            return text, model
    return None, None


def embed_texts(texts: list[str], timeout: int = 60) -> list[list[float]] | None:
    if not ENABLE_OLLAMA_EMBEDDINGS or not texts:
        return None

    data = _post_json("/api/embed", {"model": OLLAMA_EMBED_MODEL, "input": texts}, timeout=timeout)
    embeddings = (data or {}).get("embeddings")
    if embeddings:
        return embeddings

    single_embeddings = []
    for text in texts:
        item = _post_json("/api/embeddings", {"model": OLLAMA_EMBED_MODEL, "prompt": text}, timeout=timeout)
        embedding = (item or {}).get("embedding")
        if not embedding:
            return None
        single_embeddings.append(embedding)
    return single_embeddings
