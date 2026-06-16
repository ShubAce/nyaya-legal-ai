"""
rag_agent.py — Agent 2: Validates extracted statutes against the ChromaDB
statute corpus (IPC, CrPC, Constitution, etc.).

This agent catches hallucinated statute sections — the most common failure
mode of base LLMs on Indian legal text. If the model invents "Section 498-ZZ
of the IPC", this agent will catch it because no matching section exists in
the corpus.
"""

import os
import time
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from dotenv import load_dotenv

from agents.state import LegalExtractionState

load_dotenv()

CHROMA_STATUTE_PATH = Path(os.getenv("CHROMA_STATUTE_PATH", "data/chroma_statutes"))
EMBED_MODEL         = "BAAI/bge-base-en-v1.5"

# Distance threshold — below this = match found, above = hallucination
HALLUCINATION_THRESHOLD = 0.38


@lru_cache(maxsize=1)
def _load_statute_collection():
    """Load ChromaDB statute collection once and cache."""
    logger.info(f"Loading statute ChromaDB from: {CHROMA_STATUTE_PATH}")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_STATUTE_PATH))
    collection = client.get_collection("statutes", embedding_function=ef)
    logger.success(f"Statute collection loaded: {collection.count()} chunks")
    return collection


def get_standard_act_name(act_name: str) -> str:
    act_lower = act_name.lower()
    if "penal" in act_lower or "ipc" in act_lower:
        return "Ipc"
    if "constitution" in act_lower:
        return "Constitution"
    if "criminal procedure" in act_lower or "crpc" in act_lower or "cr.p.c" in act_lower:
        return "Crpc"
    if "evidence" in act_lower or "iea" in act_lower:
        return "Evidence Act"
    if "civil procedure" in act_lower or "cpc" in act_lower or "c.p.c" in act_lower:
        return "Cpc"
    if "contract" in act_lower:
        return "Contract Act"
    if "limitation" in act_lower:
        return "Limitation Act"
    if "negotiable" in act_lower:
        return "Negotiable Instruments Act"
    return act_name.title()


def _validate_statute(collection, act: str, section: str) -> dict:
    """
    Query the statute corpus for a specific act + section.
    Returns a result dict with verified flag and matched text.
    """
    query = f"{act} Section {section}"
    try:
        standard_act = get_standard_act_name(act)
        known_acts = {
            "Ipc", "Constitution", "Crpc", "Evidence Act", "Cpc",
            "Contract Act", "Limitation Act", "Negotiable Instruments Act"
        }
        
        where_clause = None
        if standard_act in known_acts:
            where_clause = {"act": standard_act}

        results = collection.query(
            query_texts=[query],
            n_results=3,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return {
                "verified":    False,
                "distance":    1.0,
                "reason":      "no_results_returned",
                "matched_text": None,
            }

        best_distance = results["distances"][0][0]
        best_doc      = results["documents"][0][0]
        best_meta     = results["metadatas"][0][0]

        verified = best_distance < HALLUCINATION_THRESHOLD

        return {
            "verified":    verified,
            "distance":    round(best_distance, 4),
            "matched_act": best_meta.get("act", ""),
            "matched_sec": best_meta.get("section", ""),
            "matched_text": best_doc[:300] if verified else None,
            "reason":      "match_found" if verified else "no_close_match",
        }

    except Exception as e:
        logger.warning(f"Statute validation query failed for '{query}': {e}")
        return {"verified": False, "distance": 1.0, "reason": f"query_error: {e}", "matched_text": None}


def rag_validation_agent(state: LegalExtractionState) -> LegalExtractionState:
    """
    Agent 2: Validate each statute in raw_extraction against the ChromaDB corpus.

    Writes to state:
        - validated_statutes: list of statutes confirmed in corpus
        - hallucinated_statutes: list of statutes NOT found in corpus
        - audit_trail: appends validation event
    """
    audit  = state.get("audit_trail", [])
    errors = state.get("errors", [])

    extraction = state.get("raw_extraction")
    if not extraction:
        audit.append({"agent": "statute_validation", "status": "skipped", "reason": "no_extraction"})
        return {**state, "validated_statutes": [], "hallucinated_statutes": [], "audit_trail": audit}

    statutes_claimed = extraction.get("statutes_cited", [])
    if not statutes_claimed:
        audit.append({
            "agent":  "statute_validation",
            "status": "no_statutes_to_validate",
        })
        return {**state, "validated_statutes": [], "hallucinated_statutes": [], "audit_trail": audit}

    try:
        collection = _load_statute_collection()
    except Exception as e:
        error_msg = f"rag_agent: could not load statute collection: {e}"
        logger.warning(error_msg)
        errors.append(error_msg)
        # Graceful degradation — mark all as unverified rather than crashing
        unverified = [{**s, "verified": False, "reason": "collection_unavailable"} for s in statutes_claimed]
        audit.append({"agent": "statute_validation", "status": "collection_unavailable", "error": str(e)})
        return {**state, "validated_statutes": [], "hallucinated_statutes": unverified,
                "audit_trail": audit, "errors": errors}

    validated, hallucinated = [], []
    t0 = time.time()

    for statute in statutes_claimed:
        act     = statute.get("act", "")
        section = statute.get("section", "")

        if not act or not section:
            # Can't validate without act + section
            hallucinated.append({**statute, "verified": False, "reason": "missing_act_or_section"})
            continue

        result = _validate_statute(collection, act, section)

        enriched = {
            **statute,
            "verified":    result["verified"],
            "distance":    result.get("distance"),
            "matched_act": result.get("matched_act"),
            "matched_sec": result.get("matched_sec"),
            "actual_text": result.get("matched_text"),
            "reason":      result.get("reason"),
        }

        if result["verified"]:
            validated.append(enriched)
        else:
            hallucinated.append(enriched)
            logger.warning(
                f"Potential hallucination: {act} §{section} "
                f"(distance={result.get('distance', '?'):.3f})"
            )

    elapsed = time.time() - t0
    total   = len(statutes_claimed)

    audit.append({
        "agent":              "statute_validation",
        "status":             "complete",
        "total_statutes":     total,
        "verified_count":     len(validated),
        "hallucinated_count": len(hallucinated),
        "hallucination_rate": round(len(hallucinated) / total, 3) if total else 0.0,
        "latency":            round(elapsed, 2),
    })

    logger.info(
        f"Statute validation: {len(validated)}/{total} verified, "
        f"{len(hallucinated)} flagged as potential hallucinations ({elapsed:.1f}s)"
    )

    return {
        **state,
        "validated_statutes":    validated,
        "hallucinated_statutes": hallucinated,
        "audit_trail":           audit,
        "errors":                errors,
    }
