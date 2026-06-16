"""
state.py — Shared state schema for the Nyaya LangGraph pipeline.

Every agent reads from and writes to this state object.
TypedDict enforces the contract between agents.
"""

from typing import Optional, TypedDict


class LegalExtractionState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    judgment_text: str

    # ── Agent 1: Extraction (Nyaya-7B) ────────────────────────────────────────
    raw_extraction: Optional[dict]          # raw JSON from finetuned model

    # ── Agent 2: Statute Validation (RAG) ─────────────────────────────────────
    validated_statutes:    Optional[list]   # statutes confirmed in corpus
    hallucinated_statutes: Optional[list]   # statutes NOT found in corpus

    # ── Agent 3: Precedent Resolution ─────────────────────────────────────────
    resolved_precedents: Optional[list]     # citations resolved to case names

    # ── Agent 4: Confidence Scoring ───────────────────────────────────────────
    per_field_confidence: Optional[dict]    # {"case_name": 0.97, "statutes_cited": 0.81, ...}
    uncertain_fields:     Optional[list]    # fields below confidence threshold (< 0.70)
    overall_confidence:   Optional[float]   # mean confidence across all fields

    # ── Final output ──────────────────────────────────────────────────────────
    final_output:  Optional[dict]           # fully enriched, validated extraction
    audit_trail:   list                     # full per-agent reasoning trace

    # ── Error handling ────────────────────────────────────────────────────────
    errors: Optional[list]                  # any non-fatal errors during processing
