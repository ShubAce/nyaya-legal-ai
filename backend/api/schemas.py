"""api/schemas.py — Pydantic schemas for all API request/response models."""

from typing import Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    judgment_text: str
    stream: bool = True


class StatuteCitationResponse(BaseModel):
    act: str
    section: str
    description: Optional[str] = None
    verified: bool
    actual_text: Optional[str] = None


class PrecedentResponse(BaseModel):
    citation: str
    case_name: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    resolved: bool
    source: str
    summary: Optional[str] = None


class AnalysisResult(BaseModel):
    case_name: str
    citation: Optional[str] = None
    court: str
    bench: Optional[list[str]] = None
    year: Optional[int] = None
    petitioner: str
    respondent: str
    subject_matter: Optional[str] = None
    legal_issues: list[str]
    holding: str
    outcome: str
    statutes_cited: list[StatuteCitationResponse]
    precedents_cited: list[PrecedentResponse]
    hallucinated_statutes: list[str]
    confidence_scores: dict[str, float]
    overall_confidence: float
    uncertain_fields: list[str]
    needs_human_review: bool


class StreamEvent(BaseModel):
    node: str
    status: str
    overall_confidence: Optional[float] = None
    final_output: Optional[dict] = None
    errors: list[str] = []


class CompareResult(BaseModel):
    judgment_text_preview: str
    systems: dict[str, dict]


class SimilarCase(BaseModel):
    citation: str
    case_name: str
    court: str
    year: int
    similarity_score: float
    holding_preview: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    statute_index_size: int
    precedent_index_size: int
