from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytesseract


def extract_image_text(path: str | Path) -> str:
    try:
        image = Image.open(path)
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""
