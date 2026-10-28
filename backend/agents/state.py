"""
state.py — Shared state schema for the Nyaya LangGraph pipeline.

Each agent returns only the keys it produces. `audit_trail` and `errors` use an
`operator.add` reducer, so agents return just their new entries and LangGraph
appends them — this is what lets the statute and precedent agents run in
parallel without overwriting each other's updates.
"""

import operator
from typing import Annotated, Optional, TypedDict


class LegalExtractionState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    judgment_text: str

    # ── Agent 1: Extraction (Nyaya-7B) ────────────────────────────────────────
    raw_extraction: Optional[dict]          # normalised JSON from the finetuned model

    # ── Agent 2: Statute Validation ───────────────────────────────────────────
    validated_statutes:    Optional[list]   # statutes found in the judgment text
    hallucinated_statutes: Optional[list]   # statutes NOT found in the judgment text

    # ── Agent 3: Precedent Resolution ─────────────────────────────────────────
    resolved_precedents: Optional[list]     # citations, resolved to case names where possible

    # ── Agent 4: Confidence Scoring ───────────────────────────────────────────
    per_field_confidence: Optional[dict]    # {"case_name": 0.97, "statutes_cited": 0.81, ...}
    uncertain_fields:     Optional[list]    # fields below confidence threshold (< 0.70)
    overall_confidence:   Optional[float]   # mean confidence across all fields

    # ── Final output ──────────────────────────────────────────────────────────
    final_output: Optional[dict]                    # fully enriched, validated extraction
    audit_trail:  Annotated[list, operator.add]     # per-agent reasoning trace

    # ── Error handling ────────────────────────────────────────────────────────
    errors: Annotated[list, operator.add]           # non-fatal errors during processing
