"""
citation_agent.py — Agent 3: Resolve Indian legal citations (AIR, SCC, SCR, etc.)
to full case names and summaries using the local ChromaDB precedent index.

Citations are matched on their parsed (reporter, year, volume, page) parts, so
"AIR 1987 SC 943" and "1987 AIR 943" resolve to the same case; case names are
fuzzy-matched as a fallback. The IndianKanoon API is a paid online service and
is only queried when USE_INDIANKANOON=1 is set in backend/.env.
"""

import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

import httpx
from loguru import logger

from agents.config import INDIANKANOON_API_KEY, USE_INDIANKANOON, get_collection
from agents.state import LegalExtractionState

# Citations found in the raw text that the model may have missed.
CITATION_PATTERNS = [
    r"AIR\s+\d{4}\s+[A-Z][A-Za-z]*\s+\d+",       # AIR 1984 SC 1622
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",              # (2019) 5 SCC 1
    r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",              # 2019 (5) SCC 1
    r"\[\d{4}\]\s+\d+\s+SCR\s+\d+",              # [1984] 2 SCR 622
    r"\d{4}\s+SCR\s+(?:\(\d+\)\s+)?\d+",         # 1984 SCR 622 / 1987 SCR (2) 444
    r"\d{4}\s+Cri\s*LJ\s+\d+",                   # 2001 Cri LJ 1234
]
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CITATION_PATTERNS]

# Parsers that turn any citation format into a comparable key.
_KEY_PATTERNS = [
    (re.compile(r"AIR\s*(\d{4})\s*[A-Z][A-Za-z.]*\s*(\d+)", re.I),     lambda m: ("AIR", m[1], m[2])),
    (re.compile(r"(\d{4})\s*AIR\s*(\d+)", re.I),                       lambda m: ("AIR", m[1], m[2])),
    (re.compile(r"\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)", re.I),           lambda m: ("SCC", m[1], m[2], m[3])),
    (re.compile(r"(\d{4})\s*\((\d+)\)\s*SCC\s*(\d+)", re.I),           lambda m: ("SCC", m[1], m[2], m[3])),
    (re.compile(r"(\d{4})\s*SCC\s*\((\d+)\)\s*(\d+)", re.I),           lambda m: ("SCC", m[1], m[2], m[3])),
    (re.compile(r"\[(\d{4})\]\s*\d+\s*SCR\s*(\d+)", re.I),             lambda m: ("SCR", m[1], m[2])),
    (re.compile(r"(\d{4})\s*SCR\s*(?:\(\d+\)\s*)?(\d+)", re.I),        lambda m: ("SCR", m[1], m[2])),
    (re.compile(r"(\d{4})\s*Cri\s*\.?\s*L\.?\s*J\.?\s*(\d+)", re.I),   lambda m: ("CRILJ", m[1], m[2])),
]


def citation_keys(text: str) -> set[tuple]:
    keys = set()
    for pattern, to_key in _KEY_PATTERNS:
        keys.update(to_key(m) for m in pattern.finditer(text))
    return keys


def _normalize_name(name: str) -> str:
    name = re.sub(r"\b(versus|vs\.?|v\.?)\b", " v ", name.lower())
    name = re.sub(r"\b(and|&)?\s*(anr|ors|others|another)\b\.?", " ", name)
    return re.sub(r"[^a-z ]+", " ", re.sub(r"\s+", " ", name)).strip()


@lru_cache(maxsize=1)
def _precedent_index() -> tuple[dict, list]:
    """Load every precedent once: ({citation_key: case}, [(normalised_name, case)])."""
    collection = get_collection("precedents")
    if collection is None:
        return {}, []
    data = collection.get(include=["documents", "metadatas"])
    by_key, by_name = {}, []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        case = {
            "case_name": meta.get("case_name", ""),
            "court":     meta.get("court", ""),
            "year":      meta.get("year", 0),
            "outcome":   meta.get("outcome", ""),
            "summary":   (doc or "")[:300],
            "citation":  meta.get("citation", ""),
        }
        for key in citation_keys(case["citation"]):
            by_key[key] = case
        if case["case_name"]:
            by_name.append((_normalize_name(case["case_name"]), case))
    return by_key, by_name


def _lookup_local_index(citation: str, case_name: str) -> Optional[dict]:
    by_key, by_name = _precedent_index()
    for key in citation_keys(citation):
        if key in by_key:
            return {"resolved": True, "source": "local_index", **by_key[key]}

    wanted = _normalize_name(case_name)
    if len(wanted) >= 8:
        best_score, best = max(((SequenceMatcher(None, wanted, name).ratio(), case)
                                for name, case in by_name), default=(0.0, None), key=lambda x: x[0])
        if best_score >= 0.85:
            return {"resolved": True, "source": "local_index_name", **best}
    return None


def _lookup_indiankanoon(citation: str) -> Optional[dict]:
    """Query the IndianKanoon search API (only when USE_INDIANKANOON=1)."""
    if not USE_INDIANKANOON:
        return None
    try:
        r = httpx.post(
            "https://api.indiankanoon.org/search/",
            headers={"Authorization": f"Token {INDIANKANOON_API_KEY}"},
            data={"formInput": citation, "pagenum": 0},
            timeout=8.0,
        )
        if r.status_code != 200:
            return None
        docs = r.json().get("docs", [])
        if not docs:
            return None
        first = docs[0]
        publish_date = first.get("publishdate") or ""
        return {
            "resolved":  True,
            "source":    "indiankanoon_api",
            "case_name": re.sub(r"<[^>]+>", "", first.get("title", "")),
            "court":     first.get("docsource", ""),
            "year":      int(publish_date[:4]) if publish_date[:4].isdigit() else 0,
            "outcome":   "",
            "summary":   re.sub(r"<[^>]+>", "", first.get("headline", ""))[:300],
            "ik_doc_id": first.get("tid", ""),
        }
    except Exception as e:
        logger.debug(f"IndianKanoon lookup failed for '{citation}': {e}")
        return None


def _extract_citations_from_text(text: str) -> list[str]:
    found = []
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            cit = re.sub(r"\s+", " ", match.group()).strip()
            if cit not in found:
                found.append(cit)
    return found


def _find_citation_near_name(case_name: str, judgment_text: str) -> str:
    """The model sometimes returns a case name but no citation; look right after the name in the text."""
    first_party = re.split(r"\s+(?:v\.?|vs\.?|versus)\s+", case_name, maxsplit=1, flags=re.I)[0].strip()
    if len(first_party) < 4:
        return ""
    idx = re.sub(r"\s+", " ", judgment_text).lower().find(first_party.lower())
    if idx == -1:
        return ""
    window = re.sub(r"\s+", " ", judgment_text)[idx: idx + len(case_name) + 150]
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(window)
        if match:
            return match.group().strip()
    return ""


def _is_grounded(citation: str, text_keys: set, text_normalized: str) -> bool:
    """True if the citation (or its reporter/year/page) appears in the judgment."""
    keys = citation_keys(citation)
    if keys:
        return bool(keys & text_keys)
    return re.sub(r"\s+", " ", citation.lower()).strip() in text_normalized


def citation_resolution_agent(state: LegalExtractionState) -> dict:
    """
    Agent 3: Resolve each precedent citation to a full case name + summary.

    Resolution strategy (in order):
        1. Local precedent index — citation match, then fuzzy case-name match
        2. IndianKanoon API (opt-in)
        3. Unresolved (kept with the model's data)

    Also adds citations found in the raw text that the model missed, and marks
    whether each citation actually appears in the judgment (`grounded`).
    """
    extraction = state.get("raw_extraction") or {}
    judgment_text = state.get("judgment_text") or ""
    text_keys = citation_keys(judgment_text)
    text_normalized = re.sub(r"\s+", " ", judgment_text.lower())

    citations = [dict(c) for c in extraction.get("precedents_cited") or []]
    for cite in citations:
        if not cite.get("citation") and cite.get("case_name"):
            cite["citation"] = _find_citation_near_name(cite["case_name"], judgment_text)
    seen_keys = set().union(*(citation_keys(c.get("citation", "")) for c in citations))
    seen_raw = {c.get("citation", "") for c in citations}

    extra_found = 0
    for cit in _extract_citations_from_text(judgment_text):
        keys = citation_keys(cit)
        if (keys and keys & seen_keys) or (not keys and cit in seen_raw):
            continue
        citations.append({"citation": cit, "case_name": "", "from_text_scan": True})
        seen_keys |= keys
        extra_found += 1

    precedents = []
    t0 = time.time()
    for cite in citations:
        citation_str = (cite.get("citation") or "").strip()
        model_name = cite.get("case_name") or ""
        if not citation_str and not model_name:
            continue

        base = {
            "original_citation": citation_str,
            "model_case_name":   model_name,
            "grounded":          _is_grounded(citation_str, text_keys, text_normalized) if citation_str else False,
            "from_text_scan":    cite.get("from_text_scan", False),
        }
        result = _lookup_local_index(citation_str, model_name) or (
            _lookup_indiankanoon(citation_str) if citation_str else None
        )
        if result:
            precedents.append({**base, **result})
        else:
            precedents.append({**base, "resolved": False, "source": "unresolved"})

    resolved_count = sum(1 for p in precedents if p["resolved"])
    elapsed = round(time.time() - t0, 2)
    logger.info(f"Citation resolution: {resolved_count}/{len(precedents)} resolved in {elapsed:.1f}s")

    return {
        "resolved_precedents": precedents,
        "audit_trail": [{
            "agent":            "citation_resolution",
            "status":           "complete",
            "total_citations":  len(precedents),
            "resolved_count":   resolved_count,
            "unresolved_count": len(precedents) - resolved_count,
            "grounded_count":   sum(1 for p in precedents if p["grounded"]),
            "resolution_rate":  round(resolved_count / len(precedents), 3) if precedents else 0.0,
            "extra_found":      extra_found,
            "indiankanoon":     USE_INDIANKANOON,
            "latency":          elapsed,
        }],
    }
