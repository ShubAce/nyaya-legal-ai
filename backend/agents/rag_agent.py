"""
rag_agent.py — Agent 2: Validates extracted statutes.

A statute is a hallucination when the judgment never mentions it, so each
extracted section is first checked against the judgment text itself. Sections
that are grounded are then looked up in the ChromaDB statute corpus (exact
"Section N." / "Article N." match) to attach the actual statutory text.

Semantic similarity is not used for verification: embeddings cannot tell
"Section 302" from "Section 307", so a nearest-neighbour distance says nothing
about whether a section number is real.
"""

import re
import time
from typing import Optional

from loguru import logger

from agents.config import get_collection
from agents.state import LegalExtractionState

# (pattern, canonical act name, act key in the statute corpus or None).
# The corpus key is None where data/statute_corpus holds the wrong document:
# crpc.txt is the 1980 amendment act, evidence_act.txt is the Bankers' Books
# Evidence Act and limitation_act.txt is the Delimitation Act.
ACTS = [
    (r"penal code|\bi\.?\s?p\.?\s?c\b",                    "Indian Penal Code",          "Ipc"),
    (r"criminal procedure|\bcr\.?\s?p\.?\s?c\b",           "Code of Criminal Procedure", None),
    (r"civil procedure|\bc\.?\s?p\.?\s?c\b",               "Code of Civil Procedure",    "Cpc"),
    (r"constitution",                                      "Constitution of India",      "Constitution"),
    (r"negotiable instruments?|\bn\.?\s?i\.?\s?act\b",     "Negotiable Instruments Act", "Negotiable Instruments Act"),
    (r"contract act",                                      "Indian Contract Act",        "Contract Act"),
    (r"\bevidence act\b",                                  "Indian Evidence Act",        None),
]

_SECTION_TOKEN = re.compile(r"\b\d{1,4}[A-Z]{0,3}\b", re.IGNORECASE)


def canonical_act(act: str) -> tuple[str, Optional[str]]:
    """Return (display name, corpus key) for an act name as written by the model."""
    lowered = act.lower()
    for pattern, name, corpus_key in ACTS:
        if re.search(pattern, lowered):
            return name, corpus_key
    return act.strip(), None


def split_sections(raw: str) -> list[str]:
    """'302/34', 'Sections 302 and 34', '498-A', '21(1)(a)' -> ['302', '34'], ['498A'], ['21']."""
    text = re.sub(r"\([^)]*\)", " ", raw)                  # drop sub-clauses like (1)(a)
    text = re.sub(r"(\d)\s*-\s*([A-Z])\b", r"\1\2", text, flags=re.IGNORECASE)
    return [s.upper() for s in _SECTION_TOKEN.findall(text)]


def normalize_text(text: str) -> str:
    """Lower-case, collapse whitespace (PDF line breaks) and join '498-A' into '498a'."""
    text = re.sub(r"\s+", " ", text.lower())
    return re.sub(r"(\d)\s*-\s*([a-z])\b", r"\1\2", text)


def mentioned_in_text(section: str, text: str, is_article: bool) -> bool:
    """True if the (normalised) judgment mentions this section/article number in a statutory context."""
    num = re.escape(section.lower())
    keyword = r"(?:articles?|art\.)" if is_article else r"(?:sections?|secs?\.?|ss?\.|u/ss?\.?|under|read with|r/w)"
    before = rf"{keyword}.{{0,80}}?(?<![\w.]){num}(?![\w])"
    after  = rf"(?<![\w.]){num}(?![\w]).{{0,25}}?(?:ipc|i\.p\.c|cr\.?p\.?c|c\.p\.c|cpc|of the|constitution|act)"
    return re.search(before, text) is not None or re.search(after, text) is not None


def lookup_corpus(collection, corpus_key: str, section: str) -> Optional[str]:
    """Exact lookup of a section's text in the statute corpus."""
    header = f"{'Article' if corpus_key == 'Constitution' else 'Section'} {section}."
    try:
        result = collection.get(
            where={"act": corpus_key},
            where_document={"$contains": header},
            include=["documents"],
            limit=10,
        )
    except Exception as e:
        logger.warning(f"Statute corpus lookup failed for {corpus_key} {header}: {e}")
        return None
    for doc in result.get("documents") or []:
        # Only accept chunks where the section heading starts a line, not cross-references.
        if re.search(rf"(^|\n)\s*{re.escape(header)}", doc):
            return doc[:400]
    return None


def rag_validation_agent(state: LegalExtractionState) -> dict:
    """
    Agent 2: Validate each statute in raw_extraction.

    Every statute gets a `status`:
        verified          — mentioned in the judgment and found in the statute corpus
        grounded          — mentioned in the judgment; the corpus does not cover it
        incomplete        — the model gave no section number, so it cannot be checked
        not_in_judgment   — the judgment never mentions it (hallucination)
    """
    extraction = state.get("raw_extraction") or {}
    statutes_claimed = extraction.get("statutes_cited") or []
    if not statutes_claimed:
        return {
            "validated_statutes": [], "hallucinated_statutes": [],
            "audit_trail": [{"agent": "statute_validation", "status": "no_statutes_to_validate"}],
        }

    judgment_text = normalize_text(state.get("judgment_text") or "")
    collection = get_collection("statutes")
    validated, hallucinated = [], []
    t0 = time.time()

    for statute in statutes_claimed:
        act_name, corpus_key = canonical_act(statute.get("act") or "")
        sections = split_sections(statute.get("section") or "")
        if not sections:
            validated.append({**statute, "act": act_name, "verified": False, "status": "incomplete"})
            continue

        is_article = act_name == "Constitution of India"
        for section in sections:
            entry = {**statute, "act": act_name, "section": section}
            if not mentioned_in_text(section, judgment_text, is_article):
                hallucinated.append({**entry, "verified": False, "status": "not_in_judgment"})
                logger.warning(f"Possible hallucination: {act_name} §{section} is not mentioned in the judgment")
                continue

            actual_text = lookup_corpus(collection, corpus_key, section) if collection and corpus_key else None
            validated.append({
                **entry,
                "verified":    True,
                "status":      "verified" if actual_text else "grounded",
                "actual_text": actual_text,
            })

    total = len(validated) + len(hallucinated)
    elapsed = round(time.time() - t0, 2)
    logger.info(f"Statute validation: {len(validated)}/{total} grounded, {len(hallucinated)} flagged ({elapsed:.1f}s)")

    return {
        "validated_statutes":    validated,
        "hallucinated_statutes": hallucinated,
        "audit_trail": [{
            "agent":              "statute_validation",
            "status":             "complete",
            "total_statutes":     total,
            "in_corpus_count":    sum(1 for s in validated if s["status"] == "verified"),
            "grounded_count":     sum(1 for s in validated if s["verified"]),
            "hallucinated_count": len(hallucinated),
            "hallucination_rate": round(len(hallucinated) / total, 3) if total else 0.0,
            "corpus_available":   collection is not None,
            "latency":            elapsed,
        }],
    }
