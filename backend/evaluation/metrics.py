"""
metrics.py — Phase 5: All evaluation metrics for the Nyaya-7B benchmark.

Metrics:
  - statute_f1:          F1 on extracted statute (act, section) pairs
  - outcome_accuracy:    exact match on case outcome classification
  - party_accuracy:      fuzzy match on petitioner + respondent
  - json_validity:       fraction of outputs that are parseable JSON
  - hallucination_rate:  fraction of cited statutes not found in corpus
  - field_coverage:      fraction of required fields present in output
"""

import json
import re
from typing import Optional

from fuzzywuzzy import fuzz
from loguru import logger


# ── Statute F1 ────────────────────────────────────────────────────────────────

def _normalize_statute(statute: dict) -> str:
    """Normalize a statute dict to a canonical string for comparison."""
    act     = (statute.get("act") or "").lower().strip()
    section = re.sub(r"[^\w]", "", (statute.get("section") or "").lower().strip())

    # Normalize common abbreviations
    act = act.replace("ipc", "indian penal code")
    act = act.replace("crpc", "code of criminal procedure")
    act = act.replace("cpc",  "code of civil procedure")
    act = act.replace("iea",  "indian evidence act")

    return f"{act}_{section}"


def statute_f1(predicted: dict, ground_truth: dict) -> dict:
    """
    Compute precision, recall, and F1 on statute citations.

    Args:
        predicted:    model output dict (has "statutes_cited")
        ground_truth: ground truth label dict
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    pred_statutes = {
        _normalize_statute(s)
        for s in (predicted.get("statutes_cited") or [])
        if isinstance(s, dict)
    }
    true_statutes = {
        _normalize_statute(s)
        for s in (ground_truth.get("statutes_cited") or [])
        if isinstance(s, dict)
    }

    if not pred_statutes and not true_statutes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_statutes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not true_statutes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    tp        = len(pred_statutes & true_statutes)
    precision = tp / len(pred_statutes)
    recall    = tp / len(true_statutes)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ── Outcome accuracy ──────────────────────────────────────────────────────────

OUTCOME_ALIASES = {
    "appeal dismissed":     "dismissed",
    "petition dismissed":   "dismissed",
    "writ petition dismissed": "dismissed",
    "appeal allowed":       "allowed",
    "petition allowed":     "allowed",
    "writ petition allowed": "allowed",
    "disposed of":          "disposed",
    "disposed of as infructuous": "disposed",
    "set aside":            "allowed",
    "confirmed":            "dismissed",
    "upheld":               "dismissed",
}


def outcome_accuracy(predicted: dict, ground_truth: dict) -> bool:
    """Exact match on outcome, after normalization and alias resolution."""
    def normalize(s: str) -> str:
        s = s.lower().strip()
        return OUTCOME_ALIASES.get(s, s)

    pred = normalize(predicted.get("outcome") or "")
    true = normalize(ground_truth.get("outcome") or "")
    return pred == true


# ── Party name accuracy ───────────────────────────────────────────────────────

def party_accuracy(predicted: dict, ground_truth: dict) -> float:
    """
    Fuzzy match on petitioner and respondent names.
    Uses token_set_ratio to handle name ordering differences.
    Returns mean of petitioner_score and respondent_score.
    """
    def fuzzy_score(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0

    pet_score  = fuzzy_score(predicted.get("petitioner") or "", ground_truth.get("petitioner") or "")
    resp_score = fuzzy_score(predicted.get("respondent") or "", ground_truth.get("respondent") or "")
    return round((pet_score + resp_score) / 2, 4)


# ── JSON validity ─────────────────────────────────────────────────────────────

def json_validity(raw_output: str) -> bool:
    """Check if model output string is parseable JSON."""
    if not raw_output:
        return False
    try:
        json.loads(raw_output.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        # Try stripping markdown fences
        cleaned = re.sub(r"```(?:json)?|```", "", raw_output).strip()
        try:
            json.loads(cleaned)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


# ── Hallucination rate ────────────────────────────────────────────────────────

def hallucination_rate(predicted: dict, ground_truth: dict) -> float:
    """
    Fraction of predicted statutes that do NOT appear in ground truth.
    A hallucinated statute is one the model invented — not in the actual judgment.
    """
    pred_statutes = {
        _normalize_statute(s)
        for s in (predicted.get("statutes_cited") or [])
        if isinstance(s, dict)
    }
    true_statutes = {
        _normalize_statute(s)
        for s in (ground_truth.get("statutes_cited") or [])
        if isinstance(s, dict)
    }

    if not pred_statutes:
        return 0.0

    hallucinated = pred_statutes - true_statutes
    return round(len(hallucinated) / len(pred_statutes), 4)


# ── Field coverage ────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "case_name", "court", "petitioner", "respondent",
    "statutes_cited", "legal_issues", "holding", "outcome"
]


def field_coverage(predicted: dict) -> float:
    """Fraction of required fields that are non-empty in the prediction."""
    covered = 0
    for field in REQUIRED_FIELDS:
        val = predicted.get(field)
        if val and val not in ("", "null", None, [], {}):
            covered += 1
    return round(covered / len(REQUIRED_FIELDS), 4)


# ── Aggregate scoring ─────────────────────────────────────────────────────────

def compute_all_metrics(
    predicted_raw: str,
    predicted:     dict,
    ground_truth:  dict,
    inference_cost_usd: float = 0.0,
) -> dict:
    """
    Compute all metrics for a single sample.

    Args:
        predicted_raw: raw string output from the model (for JSON validity check)
        predicted:     parsed prediction dict
        ground_truth:  ground truth label dict
        inference_cost_usd: API cost if applicable
    """
    sf = statute_f1(predicted, ground_truth)
    return {
        "statute_precision":   sf["precision"],
        "statute_recall":      sf["recall"],
        "statute_f1":          sf["f1"],
        "outcome_correct":     outcome_accuracy(predicted, ground_truth),
        "party_accuracy":      party_accuracy(predicted, ground_truth),
        "json_valid":          json_validity(predicted_raw),
        "hallucination_rate":  hallucination_rate(predicted, ground_truth),
        "field_coverage":      field_coverage(predicted),
        "cost_usd":            inference_cost_usd,
    }


def aggregate_metrics(per_sample_results: list[dict]) -> dict:
    """Aggregate per-sample metric dicts into mean values."""
    if not per_sample_results:
        return {}

    numeric_keys = [
        "statute_precision", "statute_recall", "statute_f1",
        "party_accuracy", "hallucination_rate", "field_coverage", "cost_usd"
    ]
    bool_keys = ["outcome_correct", "json_valid"]

    aggregated = {}
    n = len(per_sample_results)

    for key in numeric_keys:
        values = [r[key] for r in per_sample_results if key in r]
        aggregated[key] = round(sum(values) / len(values), 4) if values else 0.0

    for key in bool_keys:
        values = [r[key] for r in per_sample_results if key in r]
        aggregated[key] = round(sum(values) / len(values), 4) if values else 0.0

    aggregated["n_samples"] = n
    return aggregated
