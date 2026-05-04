from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    BaseModel = object

    def Field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return field(default_factory=default_factory)
        return field(default=default)

from medmind.config import DISCLAIMER


def _fallback_model_dump(self):
    return asdict(self)


if BaseModel is object:
    model_decorator = dataclass
else:
    model_decorator = lambda cls: cls


@model_decorator
class EvidenceSource(BaseModel):
    title: str
    source_type: str
    snippet: str
    url: str | None = None
    score: float = 0.0

    if BaseModel is object:
        model_dump = _fallback_model_dump


@model_decorator
class PossibleCondition(BaseModel):
    name: str
    likelihood: Literal["low", "medium", "high"] = "low"
    evidence: list[str] = Field(default_factory=list)

    if BaseModel is object:
        model_dump = _fallback_model_dump


@model_decorator
class HealthInsight(BaseModel):
    possible_conditions: list[PossibleCondition] = Field(default_factory=list)
    risk_level: Literal["Low", "Medium", "High", "Urgent"] = "Low"
    recommendation: str = ""
    seek_medical_help: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""
    sources: list[EvidenceSource] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    metadata: dict[str, Any] = Field(default_factory=dict)

    if BaseModel is object:
        model_dump = _fallback_model_dump


@model_decorator
class LabFinding(BaseModel):
    test: str
    value: str
    status: Literal["low", "normal", "high", "unknown"]
    explanation: str

    if BaseModel is object:
        model_dump = _fallback_model_dump


@model_decorator
class ReportInsight(BaseModel):
    findings: list[LabFinding]
    risk_level: Literal["Low", "Medium", "High", "Urgent"]
    summary: str
    recommendation: str
    sources: list[EvidenceSource] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

    if BaseModel is object:
        model_dump = _fallback_model_dump


@model_decorator
class ImageInsight(BaseModel):
    modality: str = "medical_image"
    image_type: str = "unknown"
    extracted_text: str = ""
    visual_summary: str = ""
    possible_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendation: str = ""
    model_used: str = "ocr-fallback"
    disclaimer: str = DISCLAIMER

    if BaseModel is object:
        model_dump = _fallback_model_dump
