"""
citation_agent.py — Agent 3: Resolve Indian legal citations (AIR, SCC, SCR, etc.)
to full case names and summaries using the local ChromaDB precedent index
with IndianKanoon API as a fallback.
"""

import os
import re
import time
import httpx
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from dotenv import load_dotenv

from agents.state import LegalExtractionState

load_dotenv()

CHROMA_PRECEDENT_PATH  = Path(os.getenv("CHROMA_PRECEDENT_PATH", "data/chroma_precedents"))
INDIANKANOON_API_KEY   = os.getenv("INDIANKANOON_API_KEY", "")
EMBED_MODEL            = "BAAI/bge-base-en-v1.5"

# Indian legal citation regex patterns
CITATION_PATTERNS = [
    r"AIR\s+\d{4}\s+SC\s+\d+",          # AIR 1984 SC 1622
    r"AIR\s+\d{4}\s+\w+\s+\d+",         # AIR 1984 Bom 123
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",     # (2019) 5 SCC 1
    r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",     # 2019 (5) SCC 1
    r"\d{4}\s+SCR\s+\d+",               # 1984 SCR 622
    r"\d{4}\s+Cri\s*LJ\s+\d+",          # 2001 Cri LJ 1234
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CITATION_PATTERNS]


@lru_cache(maxsize=1)
def _load_precedent_collection():
    """Load ChromaDB precedent collection once and cache."""
    logger.info(f"Loading precedent ChromaDB from: {CHROMA_PRECEDENT_PATH}")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_PRECEDENT_PATH))
    try:
        collection = client.get_collection("precedents", embedding_function=ef)
        logger.success(f"Precedent collection loaded: {collection.count()} cases")
        return collection
    except Exception as e:
        logger.warning(f"Precedent collection not found: {e}. Run build_knowledge_base.py first.")
        return None


def _normalize_citation(citation: str) -> str:
    """Normalize citation string for consistent lookup."""
    return re.sub(r"\s+", " ", citation.strip()).upper()


def _lookup_local_index(collection, citation: str) -> dict | None:
    """Try exact ID lookup, then semantic search in local ChromaDB."""
    if collection is None:
        return None

    # Strategy 1: exact ID match
    normalized = _normalize_citation(citation).replace(" ", "_")[:100]
    try:
        result = collection.get(ids=[normalized], include=["documents", "metadatas"])
        if result["documents"] and result["documents"][0]:
            meta = result["metadatas"][0]
            return {
                "resolved":  True,
                "source":    "local_exact",
                "case_name": meta.get("case_name", ""),
                "court":     meta.get("court", ""),
                "year":      meta.get("year", 0),
                "outcome":   meta.get("outcome", ""),
                "summary":   result["documents"][0][:300],
            }
    except Exception:
        pass

    # Strategy 2: semantic search
    try:
        results = collection.query(
            query_texts=[citation],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        if results["distances"] and results["distances"][0] and results["distances"][0][0] < 0.25:
            meta = results["metadatas"][0][0]
            return {
                "resolved":  True,
                "source":    "local_semantic",
                "case_name": meta.get("case_name", ""),
                "court":     meta.get("court", ""),
                "year":      meta.get("year", 0),
                "outcome":   meta.get("outcome", ""),
                "summary":   results["documents"][0][0][:300],
            }
    except Exception:
        pass

    return None


def _lookup_indiankanoon(citation: str) -> dict | None:
    """Query IndianKanoon API for a citation."""
    if not INDIANKANOON_API_KEY:
        return None

    try:
        # Search endpoint
        url     = "https://api.indiankanoon.org/search/"
        headers = {"Authorization": f"Token {INDIANKANOON_API_KEY}"}
        params  = {"formInput": citation, "pagenum": 0}

        r = httpx.get(url, headers=headers, params=params, timeout=8.0)
        time.sleep(0.3)  # Rate limit

        if r.status_code != 200:
            return None

        data = r.json()
        docs = data.get("docs", [])

        if not docs:
            return None

        first = docs[0]
        return {
            "resolved":  True,
            "source":    "indiankanoon_api",
            "case_name": first.get("title", ""),
            "court":     first.get("docsource", ""),
            "year":      int(first.get("publishdate", "0000")[:4]) if first.get("publishdate") else 0,
            "outcome":   "",
            "summary":   first.get("headline", "")[:300],
            "ik_doc_id": first.get("tid", ""),
        }

    except Exception as e:
        logger.debug(f"IndianKanoon lookup failed for '{citation}': {e}")
        return None


def _extract_citations_from_text(text: str) -> list[str]:
    """Find all legal citations in text that weren't extracted by the model."""
    found = set()
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            found.add(match.group().strip())
    return list(found)


def citation_resolution_agent(state: LegalExtractionState) -> LegalExtractionState:
    """
    Agent 3: Resolve each precedent citation to a full case name + summary.

    Resolution strategy (in order):
        1. Local ChromaDB exact ID match
        2. Local ChromaDB semantic search
        3. IndianKanoon API
        4. Unresolved (kept with original data)

    Also finds additional citations in the raw text that the extraction
    model may have missed.

    Writes to state:
        - resolved_precedents: list of enriched citation objects
        - audit_trail: appends resolution event
    """
    audit  = state.get("audit_trail", [])
    errors = state.get("errors", [])

    extraction = state.get("raw_extraction", {})
    if not extraction:
        audit.append({"agent": "citation_resolution", "status": "skipped", "reason": "no_extraction"})
        return {**state, "resolved_precedents": [], "audit_trail": audit}

    citations_from_model = extraction.get("precedents_cited", [])

    # Also find citations the model may have missed
    raw_text = state.get("judgment_text", "")
    extra_citations_str = _extract_citations_from_text(raw_text)
    model_citation_strs = {c.get("citation") or "" for c in citations_from_model}

    # Add missed citations
    for cit_str in extra_citations_str:
        if cit_str not in model_citation_strs:
            citations_from_model.append({"citation": cit_str, "case_name": None})

    collection = _load_precedent_collection()
    resolved, unresolved = [], []
    t0 = time.time()

    for cite in citations_from_model:
        citation_str = (cite.get("citation") or "").strip()
        if not citation_str:
            continue

        result = (
            _lookup_local_index(collection, citation_str)
            or _lookup_indiankanoon(citation_str)
        )

        if result:
            enriched = {
                "original_citation": citation_str,
                "model_case_name":   cite.get("case_name"),
                **result,
            }
            resolved.append(enriched)
        else:
            unresolved.append({
                "original_citation": citation_str,
                "model_case_name":   cite.get("case_name"),
                "resolved":          False,
                "source":            "unresolved",
            })
            logger.debug(f"Could not resolve citation: {citation_str}")

    all_precedents = resolved + unresolved
    elapsed = time.time() - t0
    resolution_rate = len(resolved) / len(all_precedents) if all_precedents else 0.0

    audit.append({
        "agent":              "citation_resolution",
        "status":             "complete",
        "total_citations":    len(all_precedents),
        "resolved_count":     len(resolved),
        "unresolved_count":   len(unresolved),
        "resolution_rate":    round(resolution_rate, 3),
        "extra_found":        len(extra_citations_str) - len([c for c in citations_from_model
                                                              if c.get("citation") in model_citation_strs]),
        "latency":            round(elapsed, 2),
    })

    logger.info(
        f"Citation resolution: {len(resolved)}/{len(all_precedents)} resolved "
        f"({resolution_rate:.0%}) in {elapsed:.1f}s"
    )

    return {
        **state,
        "resolved_precedents": all_precedents,
        "audit_trail":         audit,
        "errors":              errors,
    }
