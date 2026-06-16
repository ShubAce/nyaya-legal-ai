"""
active_learning.py — Phase 6: Active learning loop.

Collects low-confidence predictions, re-labels them with Gemini 1.5 Pro,
and queues them for the next training round.

Usage:
    python scripts/active_learning.py --threshold 0.70 --min-samples 100
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client  = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

REVIEW_QUEUE_PATH = Path("data/review_queue.jsonl")
AL_TRAIN_PATH     = Path("data/processed/al_augmented.jsonl")
CONFIDENCE_LOG    = Path("data/confidence_log.jsonl")

# ── Collect uncertain predictions ─────────────────────────────────────────────

def log_uncertain_prediction(
    judgment_text:    str,
    raw_output:       dict,
    per_field_conf:   dict,
    uncertain_fields: list,
    overall_conf:     float,
):
    """
    Called by the API/pipeline whenever a prediction has uncertain fields.
    Appends to the confidence log for active learning.
    """
    CONFIDENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp":        datetime.now().isoformat(),
        "judgment_text":    judgment_text[:4000],
        "overall_conf":     overall_conf,
        "uncertain_fields": uncertain_fields,
        "per_field_conf":   per_field_conf,
        "raw_output":       raw_output,
    }
    with open(CONFIDENCE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Re-label with Gemini 1.5 Pro ─────────────────────────────────────────────

from finetuning.schema import SCHEMA_DESCRIPTION

RELABEL_PROMPT = """You are a precise Indian legal data extraction expert.
Re-extract the structured information from this judgment text, paying special
attention to these fields which a prior model found uncertain: {uncertain_fields}.

Return ONLY a valid JSON object matching this schema:
{schema}

Judgment text:
{text}"""


def relabel_with_gemini_pro(judgment_text: str, uncertain_fields: list) -> dict | None:
    """
    Re-label an uncertain sample using Gemini 1.5 Pro for higher quality.
    Cost: ~$0.00125/1K input tokens — 4x cheaper than GPT-4o / comparable quality.
    """
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    try:
        prompt   = RELABEL_PROMPT.format(
            uncertain_fields=", ".join(uncertain_fields),
            schema=SCHEMA_DESCRIPTION,
            text=judgment_text[:6000],
        )
        response = gemini_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()

        # Strip markdown fences if Gemini adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed on re-label: {e}")
        return None
    except Exception as e:
        logger.warning(f"Gemini Pro re-label failed: {e}")
        time.sleep(2)
        return None


# ── Format into training sample ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Nyaya, a specialized Indian legal extraction model. "
    "Given a judgment text, extract all structured information and return "
    "a single valid JSON object. Never hallucinate statute sections or citations "
    "not present in the text. Return only JSON — no preamble, no explanation."
)


def to_chatml(judgment_text: str, label: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Extract structured data from this Indian court judgment:\n\n{judgment_text[:4000]}"},
            {"role": "assistant", "content": json.dumps(label, indent=2, ensure_ascii=False)},
        ],
        "source":   "active_learning",
        "al_round": datetime.now().strftime("%Y%m%d"),
    }


# ── Main active learning loop ─────────────────────────────────────────────────

def run_active_learning_round(
    confidence_threshold: float = 0.70,
    min_samples:          int   = 100,
    max_relabel:          int   = 500,
):
    """
    1. Load low-confidence predictions from the confidence log
    2. Re-label with Gemini 1.5 Pro
    3. Append to augmented training set
    4. Report statistics
    """
    if not CONFIDENCE_LOG.exists():
        logger.warning(f"No confidence log at {CONFIDENCE_LOG}. Run the pipeline on some judgments first.")
        return

    # Load entries below threshold
    uncertain_samples = []
    with open(CONFIDENCE_LOG, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("overall_conf", 1.0) < confidence_threshold:
                uncertain_samples.append(entry)

    logger.info(f"Found {len(uncertain_samples)} uncertain predictions "
                f"(confidence < {confidence_threshold})")

    if len(uncertain_samples) < min_samples:
        logger.info(f"Need at least {min_samples} before retraining. "
                    f"Currently {len(uncertain_samples)} — accumulating...")
        return

    # Sample up to max_relabel
    if len(uncertain_samples) > max_relabel:
        import random
        random.shuffle(uncertain_samples)
        uncertain_samples = uncertain_samples[:max_relabel]

    logger.info(f"Re-labelling {len(uncertain_samples)} samples with Gemini 1.5 Pro...")

    new_samples    = []
    cost_estimate  = 0.0

    for entry in tqdm(uncertain_samples, desc="Re-labelling"):
        text             = entry["judgment_text"]
        uncertain_fields = entry.get("uncertain_fields", [])

        new_label = relabel_with_gemini_pro(text, uncertain_fields)
        if new_label is None:
            continue

        new_samples.append(to_chatml(text, new_label))
        # Gemini 1.5 Pro: $0.00125/1K input tokens
        cost_estimate += (len(text[:6000]) // 4 / 1000) * 0.00125
        time.sleep(0.2)  # gentle rate limiting

    # Append to augmented training set
    AL_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AL_TRAIN_PATH, "a", encoding="utf-8") as f:
        for sample in new_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.success(
        f"Active learning round complete:\n"
        f"  New samples added:  {len(new_samples)}\n"
        f"  Estimated cost:     ${cost_estimate:.3f}  (Gemini Pro, 4x cheaper than GPT-4o / comparable quality)\n"
        f"  Saved to:           {AL_TRAIN_PATH}\n"
        f"\nNext: retrain with train.jsonl + al_augmented.jsonl combined"
    )

    _remove_processed_entries(uncertain_samples)
    return new_samples


def _remove_processed_entries(processed: list):
    """Remove re-labelled entries from log to avoid re-processing."""
    processed_ts = {e["timestamp"] for e in processed}
    remaining    = []
    with open(CONFIDENCE_LOG, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("timestamp") not in processed_ts:
                remaining.append(line)
    with open(CONFIDENCE_LOG, "w", encoding="utf-8") as f:
        f.writelines(remaining)
    logger.info(f"Confidence log cleaned: {len(remaining)} entries remaining")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run active learning round")
    parser.add_argument("--threshold",   type=float, default=0.70)
    parser.add_argument("--min-samples", type=int,   default=100)
    parser.add_argument("--max-relabel", type=int,   default=500)
    args = parser.parse_args()
    run_active_learning_round(args.threshold, args.min_samples, args.max_relabel)
