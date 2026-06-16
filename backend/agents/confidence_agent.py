"""
confidence_agent.py — Agent 4: Per-field confidence scoring and uncertainty flagging.

This agent scores how confident the system is in each extracted field,
using a combination of:
  - Rule-based checks (known valid values, format validation)
  - RAG validation results (statute verification rate)
  - Citation resolution rate
  - Heuristic text length / completeness checks

Fields below the threshold are flagged for human review.
This is how production legal AI systems work — they don't pretend to be
100% correct, they quantify and surface their own uncertainty.
"""

import re
from loguru import logger
from agents.state import LegalExtractionState

# Threshold below which a field is flagged for human review
CONFIDENCE_THRESHOLD = 0.70

# Known valid values for controlled fields
VALID_COURTS = {
    "supreme court of india",
    "high court of allahabad", "high court of bombay", "high court of calcutta",
    "high court of delhi", "high court of gujarat", "high court of karnataka",
    "high court of kerala", "high court of madras", "high court of patna",
    "high court of punjab and haryana", "high court of rajasthan",
    "high court of andhra pradesh", "high court of telangana",
    "high court of jharkhand", "high court of chhattisgarh",
    "high court of uttarakhand", "high court of himachal pradesh",
    "high court of orissa", "high court of gauhati",
    "national green tribunal", "national consumer disputes redressal commission",
}

VALID_OUTCOMES = {
    "allowed", "dismissed", "disposed", "remanded", "modified", "withdrawn", "settled"
}

VALID_SUBJECT_MATTERS = {
    "criminal", "civil", "constitutional", "service", "tax",
    "family", "property", "labour", "company", "environmental",
    "consumer", "arbitration", "intellectual property", "election", "administrative",
}

# Indian legal citation format regex
CITATION_REGEX = re.compile(
    r"(AIR\s+\d{4}|(\d{4})\s*\(\d+\)\s*SCC|\(\d{4}\)\s*\d+\s*SCC|\d{4}\s+SCR)",
    re.IGNORECASE
)


def _score_case_name(extraction: dict) -> float:
    """Case name should be "X v. Y" format."""
    name = extraction.get("case_name") or ""
    if not name or len(name) < 5:
        return 0.0
    # Should contain "v." or "versus" or "vs."
    if re.search(r"\bv\.?\b|\bversus\b|\bvs\.?\b", name, re.IGNORECASE):
        return 0.95
    # Has something but no "v." — possible but lower confidence
    return 0.60


def _score_court(extraction: dict) -> float:
    court = extraction.get("court") or ""
    if not court:
        return 0.0
    if court.lower().strip() in VALID_COURTS:
        return 1.0
    # Partial match — contains "court"
    if "court" in court.lower():
        return 0.75
    return 0.40


def _score_year(extraction: dict, judgment_text: str) -> float:
    year = extraction.get("year")
    if year is None:
        return 0.0
    try:
        y = int(year)
        if 1950 <= y <= 2025:
            # Cross-check: does the year appear in the judgment text?
            if str(y) in judgment_text:
                return 1.0
            return 0.80
        return 0.20  # Year out of plausible range
    except (ValueError, TypeError):
        return 0.0


def _score_parties(extraction: dict) -> float:
    petitioner = extraction.get("petitioner") or ""
    respondent = extraction.get("respondent") or ""
    if not petitioner or not respondent:
        return 0.0
    if petitioner.lower() == respondent.lower():
        return 0.10  # Same party — likely hallucination
    if len(petitioner) > 3 and len(respondent) > 3:
        return 0.90
    return 0.50


def _score_outcome(extraction: dict) -> float:
    outcome = (extraction.get("outcome") or "").lower().strip()
    if not outcome:
        return 0.0
    if outcome in VALID_OUTCOMES:
        return 1.0
    # Partial matches
    for valid in VALID_OUTCOMES:
        if valid in outcome:
            return 0.75
    return 0.20


def _score_holding(extraction: dict) -> float:
    holding = extraction.get("holding") or ""
    if not holding:
        return 0.0
    if len(holding) < 20:
        return 0.30
    if len(holding) < 50:
        return 0.60
    # Good holding: mentions the outcome and some reasoning
    has_reasoning_words = any(w in holding.lower() for w in [
        "therefore", "accordingly", "held", "found", "concluded",
        "thus", "hence", "established", "proved", "satisfied"
    ])
    return 0.95 if has_reasoning_words else 0.80


def _score_statutes(validated: list, hallucinated: list) -> float:
    total = len(validated) + len(hallucinated)
    if total == 0:
        return 0.50  # No statutes — uncertain (might be legit for some cases)
    verified_rate = len(validated) / total
    # Penalise heavily if hallucination rate is high
    if verified_rate < 0.5:
        return 0.30
    return round(0.50 + verified_rate * 0.50, 3)  # scales 0.50 → 1.0


def _score_precedents(resolved: list) -> float:
    if not resolved:
        return 0.70  # No precedents cited — often fine
    resolved_count = sum(1 for r in resolved if r.get("resolved", False))
    resolution_rate = resolved_count / len(resolved)
    if resolution_rate == 0:
        return 0.40
    return round(0.50 + resolution_rate * 0.50, 3)


def _score_legal_issues(extraction: dict) -> float:
    issues = extraction.get("legal_issues", [])
    if not issues:
        return 0.40
    if len(issues) == 0:
        return 0.40
    # Issues should be questions or substantial phrases
    avg_len = sum(len(i) for i in issues) / len(issues)
    if avg_len > 20:
        return 0.90
    if avg_len > 10:
        return 0.70
    return 0.50


def _score_citation_format(extraction: dict) -> float:
    """Validate the top-level citation string format."""
    citation = extraction.get("citation") or ""
    if not citation:
        return 0.50  # Citation often not mentioned explicitly — okay
    if CITATION_REGEX.search(citation):
        return 1.0
    return 0.40


def confidence_scoring_agent(state: LegalExtractionState) -> LegalExtractionState:
    """
    Agent 4: Score per-field confidence and flag uncertain fields.

    Writes to state:
        - per_field_confidence: dict of {field_name: float 0-1}
        - uncertain_fields: list of field names below CONFIDENCE_THRESHOLD
        - overall_confidence: mean of all field scores
        - audit_trail: appends scoring event
    """
    audit  = state.get("audit_trail", [])
    errors = state.get("errors", [])

    extraction   = state.get("raw_extraction", {}) or {}
    validated    = state.get("validated_statutes", []) or []
    hallucinated = state.get("hallucinated_statutes", []) or []
    resolved     = state.get("resolved_precedents", []) or []
    judgment_text = state.get("judgment_text", "")

    per_field = {
        "case_name":       _score_case_name(extraction),
        "court":           _score_court(extraction),
        "year":            _score_year(extraction, judgment_text),
        "parties":         _score_parties(extraction),
        "outcome":         _score_outcome(extraction),
        "holding":         _score_holding(extraction),
        "statutes_cited":  _score_statutes(validated, hallucinated),
        "precedents_cited":_score_precedents(resolved),
        "legal_issues":    _score_legal_issues(extraction),
        "citation_format": _score_citation_format(extraction),
    }

    uncertain = [
        field for field, score in per_field.items()
        if score < CONFIDENCE_THRESHOLD
    ]
    overall = round(sum(per_field.values()) / len(per_field), 3)

    # Log uncertain fields as warnings
    if uncertain:
        logger.warning(f"Low-confidence fields (< {CONFIDENCE_THRESHOLD}): {uncertain}")
        for field in uncertain:
            logger.warning(f"  {field}: {per_field[field]:.2f}")

    audit.append({
        "agent":             "confidence_scoring",
        "status":            "complete",
        "overall_confidence": overall,
        "per_field":         per_field,
        "uncertain_fields":  uncertain,
        "flagged_count":     len(uncertain),
        "threshold":         CONFIDENCE_THRESHOLD,
    })

    logger.info(
        f"Confidence scoring: overall={overall:.2f}, "
        f"{len(uncertain)} fields flagged for review"
    )

    return {
        **state,
        "per_field_confidence": per_field,
        "uncertain_fields":     uncertain,
        "overall_confidence":   overall,
        "audit_trail":          audit,
        "errors":               errors,
    }
