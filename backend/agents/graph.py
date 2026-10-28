"""
graph.py — LangGraph pipeline wiring the four agents plus a final assembly node.

Pipeline:

                     ┌─> validate_statutes ──┐
    extract ── ok ───┤                       ├─> score_confidence ─> assemble ─> END
       │             └─> resolve_precedents ─┘                          ^
       └── extraction failed ───────────────────────────────────────────┘

Statute validation and precedent resolution are independent, so they run in
parallel. If the model output cannot be parsed, the graph skips straight to
assembly, which reports the failure instead of an empty "successful" result.

Usage:
    from agents.graph import build_pipeline, run_pipeline

    pipeline = build_pipeline()
    result   = run_pipeline(pipeline, judgment_text)
"""

import json
import sys
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: F401  — offline mode + paths, before any model library loads

# Import datasets first to resolve Windows OpenMP/CUDA DLL collision between PyArrow and PyTorch
import datasets  # noqa: F401

from langgraph.graph import StateGraph, START, END
from loguru import logger

from agents.state import LegalExtractionState
from agents.extraction_agent  import extraction_agent
from agents.rag_agent         import rag_validation_agent
from agents.citation_agent    import citation_resolution_agent
from agents.confidence_agent  import confidence_scoring_agent


# ── Final assembly node ───────────────────────────────────────────────────────

def assembly_agent(state: LegalExtractionState) -> dict:
    """
    Final node: assemble the complete enriched output from all agent outputs.
    Leaves final_output as None when extraction failed.
    """
    extraction = state.get("raw_extraction")
    if not extraction:
        logger.error("Assembly: nothing to assemble because extraction failed")
        return {
            "final_output": None,
            "audit_trail":  [{"agent": "assembly", "status": "failed", "reason": "no_extraction"}],
        }

    validated    = state.get("validated_statutes") or []
    hallucinated = state.get("hallucinated_statutes") or []
    resolved     = state.get("resolved_precedents") or []
    confidence   = state.get("per_field_confidence") or {}
    uncertain    = state.get("uncertain_fields") or []

    all_statutes = [
        {
            "act":         s.get("act", ""),
            "section":     s.get("section", ""),
            "description": s.get("description", ""),
            "verified":    s.get("verified", False),
            "status":      s.get("status", ""),
            "actual_text": s.get("actual_text"),
        }
        for s in (validated + hallucinated)
    ]

    enriched_precedents = [
        {
            "citation":  r.get("original_citation", ""),
            "case_name": r.get("case_name") or r.get("model_case_name", ""),
            "court":     r.get("court", ""),
            "year":      r.get("year", 0),
            "resolved":  r.get("resolved", False),
            "grounded":  r.get("grounded", False),
            "source":    r.get("source", ""),
            "summary":   r.get("summary", ""),
        }
        for r in resolved
    ]

    final_output = {
        # Core extraction fields
        "case_name":      extraction.get("case_name", ""),
        "citation":       extraction.get("citation", ""),
        "court":          extraction.get("court", ""),
        "bench":          extraction.get("bench", []),
        "year":           extraction.get("year"),
        "petitioner":     extraction.get("petitioner", ""),
        "respondent":     extraction.get("respondent", ""),
        "subject_matter": extraction.get("subject_matter", ""),
        "legal_issues":   extraction.get("legal_issues", []),
        "holding":        extraction.get("holding", ""),
        "outcome":        extraction.get("outcome", ""),
        "outcome_label":  extraction.get("outcome_label"),

        # Enriched by agents
        "statutes_cited":   all_statutes,
        "precedents_cited": enriched_precedents,

        # Quality metadata
        "hallucinated_statutes": [f"{s.get('act') or 'Unknown Act'} §{s.get('section') or '?'}" for s in hallucinated],
        "confidence_scores":     confidence,
        "overall_confidence":    state.get("overall_confidence") or 0.0,
        "uncertain_fields":      uncertain,
        "needs_human_review":    len(uncertain) > 0 or len(hallucinated) > 0,
    }

    logger.info(
        f"Assembly complete — "
        f"overall_confidence={final_output['overall_confidence']:.2f}, "
        f"hallucinations={len(hallucinated)}, "
        f"needs_review={final_output['needs_human_review']}"
    )

    return {
        "final_output": final_output,
        "audit_trail":  [{"agent": "assembly", "status": "complete", "fields": list(final_output.keys())}],
    }


# ── Graph definition ──────────────────────────────────────────────────────────

def _route_after_extraction(state: LegalExtractionState):
    if state.get("raw_extraction"):
        return ["validate_statutes", "resolve_precedents"]
    return "assemble"


def build_pipeline():
    """Build and compile the full LangGraph pipeline."""
    graph = StateGraph(LegalExtractionState)

    graph.add_node("extract",            extraction_agent)
    graph.add_node("validate_statutes",  rag_validation_agent)
    graph.add_node("resolve_precedents", citation_resolution_agent)
    graph.add_node("score_confidence",   confidence_scoring_agent)
    graph.add_node("assemble",           assembly_agent)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract", _route_after_extraction,
        ["validate_statutes", "resolve_precedents", "assemble"],
    )
    graph.add_edge(["validate_statutes", "resolve_precedents"], "score_confidence")
    graph.add_edge("score_confidence", "assemble")
    graph.add_edge("assemble", END)

    compiled = graph.compile()
    logger.info("LangGraph pipeline compiled: extract -> (statutes || precedents) -> confidence -> assemble")
    return compiled


# ── Public API ────────────────────────────────────────────────────────────────

def _initial_state(judgment_text: str) -> LegalExtractionState:
    return {"judgment_text": judgment_text, "audit_trail": [], "errors": []}


def run_pipeline(pipeline, judgment_text: str) -> dict:
    """
    Run the full pipeline on a judgment text.
    Returns the final state including final_output, audit_trail and errors.
    """
    t0 = time.time()
    result = pipeline.invoke(_initial_state(judgment_text))
    logger.info(f"Pipeline complete in {time.time() - t0:.1f}s")
    return result


def stream_pipeline(pipeline, judgment_text: str) -> Iterator[dict]:
    """
    Stream pipeline events as each agent completes.
    Yields {node, status, audit_trail, final_output, overall_confidence, errors}
    where audit_trail / errors hold only that node's new entries.
    """
    for event in pipeline.stream(_initial_state(judgment_text), stream_mode="updates"):
        for node_name, update in event.items():
            update = update or {}
            yield {
                "node":               node_name,
                "status":             "complete",
                "audit_trail":        update.get("audit_trail", []),
                "final_output":       update.get("final_output"),
                "overall_confidence": update.get("overall_confidence"),
                "errors":             update.get("errors", []),
            }


# ── CLI for quick testing ─────────────────────────────────────────────────────

SAMPLE_JUDGMENT = """
IN THE SUPREME COURT OF INDIA
Criminal Appeal No. 1481 of 2018
(Arising out of SLP (Crl.) No. 9525 of 2017)

STATE OF PUNJAB                          ...Appellant
                 Versus
GURPREET SINGH @ GOPI & ANR.             ...Respondents

JUDGMENT

J. Chelameswar, J.

The State of Punjab challenges the judgment dated 20.09.2017 passed by the
High Court of Punjab and Haryana at Chandigarh in Criminal Appeal No. 1234-SB
of 2010, whereby the High Court acquitted the respondents of the offences
punishable under Sections 302 and 34 of the Indian Penal Code.

The prosecution case, in brief, is that on 15.03.2009, the deceased Harjinder
Singh was allegedly murdered by the respondents. The Trial Court convicted the
respondents under Section 302 IPC read with Section 34 IPC and sentenced them
to undergo imprisonment for life.

The High Court, on reappreciation of the evidence on record, acquitted both
the accused on the ground that the prosecution failed to prove the case beyond
reasonable doubt.

The learned counsel for the State relied upon Sharad Birdhichand Sarda v. State
of Maharashtra, AIR 1984 SC 1622 to urge that the High Court erred in
reappreciating the evidence.

HELD: The appeal is allowed. The judgment of the High Court is set aside and
the conviction recorded by the Trial Court is restored. The case involved
eyewitness testimony supported by forensic evidence, which the High Court
erroneously discarded. The chain of evidence was complete and consistent with
the guilt of the accused.
"""


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        text = SAMPLE_JUDGMENT

    logger.info("Building and running Nyaya pipeline...")
    pipeline = build_pipeline()
    result   = run_pipeline(pipeline, text)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(json.dumps(result.get("final_output"), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("AUDIT TRAIL")
    print("=" * 60)
    for event in result["audit_trail"]:
        print(json.dumps(event, indent=2))

    if result.get("errors"):
        print("\n" + "=" * 60)
        print("ERRORS")
        print("=" * 60)
        for err in result["errors"]:
            print(f"  [ERROR]  {err}")
