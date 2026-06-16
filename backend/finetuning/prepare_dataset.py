"""
prepare_dataset.py — Phase 1: Dataset collection, Gemini labelling, and cleaning.

Run this on any machine with internet access (not Kaggle — it's for data prep).
Output: data/processed/train.jsonl and data/test_set/test.jsonl

Usage:
    python finetuning/prepare_dataset.py --max-samples 10000
"""

import os
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import httpx
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

# ── NEW Gemini SDK (google-genai, not google-generativeai) ────────────────────
from google import genai
from google.genai import types

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_SET_DIR  = DATA_DIR / "test_set"

for d in [RAW_DIR, PROCESSED_DIR, TEST_SET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Gemini key rotation pool ──────────────────────────────────────────────────
# Supports up to 9 API keys — add all you have in .env as
# GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... GEMINI_API_KEY_9
# Free tier = 15 RPM per key → 9 keys = 135 RPM = ~3600 rows/hour
import itertools
import threading
from collections import defaultdict

_raw_keys = [
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
    os.getenv("GEMINI_API_KEY_5", ""),
    os.getenv("GEMINI_API_KEY_6", ""),
    os.getenv("GEMINI_API_KEY_7", ""),
    os.getenv("GEMINI_API_KEY_8", ""),
    os.getenv("GEMINI_API_KEY_9", ""),
    # fallback: single key field for backwards compatibility
    os.getenv("GEMINI_API_KEY",   ""),
]
GEMINI_API_KEYS = list(dict.fromkeys(k for k in _raw_keys if k))  # deduplicate, preserve order

if not GEMINI_API_KEYS:
    raise RuntimeError(
        "No Gemini API keys found.\n"
        "Add GEMINI_API_KEY_1 through GEMINI_API_KEY_9 in your .env file.\n"
        "Get free keys at: https://aistudio.google.com"
    )

logger.info(f"Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")

# One client per key
_key_clients: dict[str, genai.Client] = {
    k: genai.Client(api_key=k) for k in GEMINI_API_KEYS
}

# Round-robin iterator
_key_pool = itertools.cycle(GEMINI_API_KEYS)

# Track call timestamps per key for rate limiting
_key_call_times: dict[str, list] = defaultdict(list)
_key_lock = threading.Lock()

RPM_LIMIT   = 14          # stay under 15 RPM limit per key (1 buffer)
WINDOW_SECS = 60.0        # sliding window


def _get_available_client() -> genai.Client:
    """
    Return a Gemini client whose key has not hit the RPM limit.
    Rotates through all keys round-robin. If all are exhausted,
    waits until the oldest call in any key expires.
    """
    checked = 0
    while True:
        key = next(_key_pool)
        now = time.time()
        with _key_lock:
            # Evict calls outside the 60-second window
            _key_call_times[key] = [
                t for t in _key_call_times[key] if now - t < WINDOW_SECS
            ]
            if len(_key_call_times[key]) < RPM_LIMIT:
                _key_call_times[key].append(now)
                return _key_clients[key]

        checked += 1
        if checked >= len(GEMINI_API_KEYS):
            # All keys saturated — find the shortest wait time
            with _key_lock:
                wait_times = []
                for k in GEMINI_API_KEYS:
                    if _key_call_times[k]:
                        oldest = min(_key_call_times[k])
                        wait_times.append(WINDOW_SECS - (now - oldest) + 0.5)
            wait = min(wait_times) if wait_times else 5.0
            wait = max(1.0, wait)   # at least 1 second
            logger.info(f"All {len(GEMINI_API_KEYS)} keys at {RPM_LIMIT} RPM — waiting {wait:.1f}s...")
            time.sleep(wait)
            checked = 0


INDIANKANOON_API_KEY = os.getenv("INDIANKANOON_API_KEY", "")

# ── Import schema ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from schema import SCHEMA_DESCRIPTION, OUTCOME_TO_LABEL

# ── Labeller prompt ───────────────────────────────────────────────────────────
LABELLER_PROMPT = """You are a precise Indian legal data extraction system.
Extract structured information from this Indian Supreme Court or High Court judgment.
Return ONLY a valid JSON object matching this schema exactly. No preamble, no explanation.
If a field cannot be determined, use null for optional fields.
Extract ALL statute and precedent mentions in the text.
Be careful with Indian legal citation formats: AIR, SCC, SCR, Cri LJ.

Schema:
{schema}

Judgment text:
{text}"""


# ── Step 1: Collect raw judgments ─────────────────────────────────────────────

def _extract_text(row: dict) -> str:
    """
    Try every known column name across all datasets.
    Returns the longest non-empty string field that looks like a judgment.
    """
    candidates = [
        row.get("text"),
        row.get("judgment"),
        row.get("document"),
        row.get("judgement"),        # alternate spelling
        row.get("full_text"),
        row.get("body"),
        row.get("content"),
        row.get("case_text"),
        row.get("input"),
        row.get("passage"),
        row.get("context"),
    ]
    # Also check nested: some datasets put text inside a dict
    for v in list(row.values()):
        if isinstance(v, str) and len(v) > 500:
            candidates.append(v)

    # Return longest non-None string above 500 chars
    valid = [c for c in candidates if c and isinstance(c, str) and len(c) > 500]
    return max(valid, key=len) if valid else ""


def collect_from_huggingface() -> list[dict]:
    """Load judgments from confirmed working HuggingFace datasets."""
    logger.info("Loading HuggingFace datasets...")
    all_texts = []

    # ── Dataset 1: IL-TUR/summ — 7030 Indian SC judgment + summary pairs ─────
    # split='train', document column is a LIST of sentences → join them
    try:
        count_before = len(all_texts)
        ds = load_dataset(
            "Exploration-Lab/IL-TUR", "summ",
            split="train", trust_remote_code=False
        )
        for row in ds:
            doc = row.get("document", "")
            # document is a list of sentence strings — join into full text
            if isinstance(doc, list):
                text = " ".join(doc)
            else:
                text = str(doc)
            if len(text) > 500:
                all_texts.append({
                    "text":   text,
                    "source": "IL-TUR-summ",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"IL-TUR/summ: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"IL-TUR/summ failed: {e}")

    # ── Dataset 2: IL-TUR/bail — 123K bail judgment docs ─────────────────────
    # split is 'train_all' not 'train'
    # Need to check column name — try _extract_text as fallback
    try:
        count_before = len(all_texts)
        ds = load_dataset(
            "Exploration-Lab/IL-TUR", "bail",
            split="train_all", trust_remote_code=False
        )
        logger.info(f"IL-TUR/bail columns: {ds.column_names}")
        if len(ds) > 0:
            first = ds[0]
            for k, v in first.items():
                if isinstance(v, str) and len(v) > 100:
                    logger.info(f"  [{k}] len={len(v)}: {v[:80]}")
                elif isinstance(v, list) and len(v) > 0:
                    logger.info(f"  [{k}] (list len={len(v)}): {str(v[0])[:80]}")

        for row in ds:
            # bail column name unknown — try all candidates + list join
            text = ""
            for col in ["text", "document", "judgment", "Text", "Document", "passage", "context"]:
                val = row.get(col, "")
                if isinstance(val, list):
                    val = " ".join(str(s) for s in val)
                if val and len(str(val)) > 500:
                    text = str(val)
                    break
            if text:
                all_texts.append({
                    "text":   text,
                    "source": "IL-TUR-bail",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"IL-TUR/bail: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"IL-TUR/bail failed: {e}")

    # ── Dataset 3: IL-TUR/cjpe — judgment prediction dataset ─────────────────
    # split is 'single_train'
    try:
        count_before = len(all_texts)
        ds = load_dataset(
            "Exploration-Lab/IL-TUR", "cjpe",
            split="single_train", trust_remote_code=False
        )
        logger.info(f"IL-TUR/cjpe columns: {ds.column_names}")
        if len(ds) > 0:
            first = ds[0]
            for k, v in first.items():
                if isinstance(v, (str, list)) and len(str(v)) > 100:
                    logger.info(f"  [{k}]: {str(v)[:80]}")

        for row in ds:
            for col in ["text", "document", "judgment", "Text", "facts", "context"]:
                val = row.get(col, "")
                if isinstance(val, list):
                    val = " ".join(str(s) for s in val)
                if val and len(str(val)) > 500:
                    text = str(val)
                    all_texts.append({
                        "text":   text,
                        "source": "IL-TUR-cjpe",
                        "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                    })
                    break
        logger.success(f"IL-TUR/cjpe: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"IL-TUR/cjpe failed: {e}")

    # ── Dataset 4: InJudgements — 11,970 Indian SC/HC judgments ──────────────
    # column = 'Text' (capital T) — confirmed from inspect output
    try:
        count_before = len(all_texts)
        ds = load_dataset(
            "opennyaiorg/InJudgements_dataset",
            split="train", trust_remote_code=False
        )
        for row in ds:
            text = row.get("Text", "")   # capital T — confirmed
            if text and len(text) > 500:
                all_texts.append({
                    "text":   text,
                    "source": "InJudgements",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"InJudgements: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"InJudgements failed: {e}")

    # ── Dataset 5: rishiai — 6,944 judgment + summary pairs ──────────────────
    # column = 'Judgment' (capital J) — confirmed from inspect output
    try:
        count_before = len(all_texts)
        ds = load_dataset(
            "rishiai/indian-court-judgements-and-its-summaries",
            split="train", trust_remote_code=False
        )
        for row in ds:
            text = row.get("Judgment", "")   # capital J — confirmed
            if text and len(text) > 500:
                all_texts.append({
                    "text":   text,
                    "source": "RishiAI",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"RishiAI: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"RishiAI failed: {e}")

    # nisaar/LawyerGPT skipped — only 150 rows, not worth it

    # ── IndianKanoon fallback ─────────────────────────────────────────────────
    if INDIANKANOON_API_KEY and len(all_texts) < 2000:
        logger.warning(f"Only {len(all_texts)} samples — trying IndianKanoon API")
        all_texts.extend(_fetch_indiankanoon_sample())

    logger.info(f"Total raw samples collected: {len(all_texts)}")
    return all_texts


def _fetch_indiankanoon_sample() -> list[dict]:
    """Fetch a small sample from IndianKanoon API as fallback."""
    if not INDIANKANOON_API_KEY:
        return []
    # Sample doc IDs — well-known SC judgments
    sample_ids = [
        "1951_10", "134584", "445276", "1317888", "74233",
        "1966609", "271328", "183203", "445702", "1228291",
    ]
    results = []
    for doc_id in sample_ids:
        try:
            url     = f"https://api.indiankanoon.org/doc/{doc_id}/"
            headers = {"Authorization": f"Token {INDIANKANOON_API_KEY}"}
            r       = httpx.get(url, headers=headers, timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                text = data.get("doc", "")
                if len(text) > 500:
                    results.append({
                        "text":   text,
                        "source": "IndianKanoon",
                        "id":     f"ik_{doc_id}",
                    })
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"IK fetch failed for {doc_id}: {e}")
    return results


def deduplicate(samples: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for s in samples:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)
    logger.info(f"After dedup: {len(unique)} unique (removed {len(samples) - len(unique)})")
    return unique


def round_robin_sample(samples: list[dict], max_samples: int) -> list[dict]:
    """
    Groups samples by their 'source' dataset, shuffles each group deterministically,
    and interleaves them in a round-robin fashion up to max_samples.
    This ensures balanced dataset representation and sequential round-robin labeling
    while preserving the resume capability of the script across runs.
    """
    # Group samples by source
    by_source = defaultdict(list)
    for s in samples:
        by_source[s["source"]].append(s)

    # Sort sources alphabetically to make the order of sources completely deterministic
    sources = sorted(by_source.keys())

    # Sort samples within each source by ID, then shuffle deterministically
    local_random = random.Random(42)
    for src in sources:
        by_source[src].sort(key=lambda x: x["id"])
        local_random.shuffle(by_source[src])

    # Interleave in a round-robin rotation
    interleaved = []
    indices = {src: 0 for src in sources}

    while len(interleaved) < max_samples:
        active_sources = [src for src in sources if indices[src] < len(by_source[src])]
        if not active_sources:
            break

        for src in active_sources:
            if len(interleaved) >= max_samples:
                break
            idx = indices[src]
            interleaved.append(by_source[src][idx])
            indices[src] += 1

    return interleaved


# ── Step 2: Label with Gemini Flash (rotating keys) ──────────────────────────

def label_judgment(text: str) -> Optional[dict]:
    """
    Call Gemini 1.5 Flash with automatic key rotation.

    - Rotates across all GEMINI_API_KEY_1..9 round-robin
    - Respects 15 RPM per key (stays at 14 to be safe)
    - Auto-waits when all keys are saturated
    - Retries up to 3 times on transient errors
    - Cost: ~$0.000075/1K tokens — 2000 rows ≈ $0.09 total
    """
    truncated = text[:4000]   # keep prompts short to maximise RPM throughput
    prompt    = LABELLER_PROMPT.format(schema=SCHEMA_DESCRIPTION, text=truncated)

    for attempt in range(3):
        try:
            client   = _get_available_client()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()

            # Strip markdown fences if Gemini adds them
            if raw.startswith("```"):
                parts = raw.split("```")
                raw   = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}: JSON parse failed — {e}")
            return None   # bad JSON is not retryable
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "rate" in err or "429" in err:
                wait = 15 * (attempt + 1)
                logger.warning(f"Attempt {attempt+1}: quota hit — waiting {wait}s")
                time.sleep(wait)
            elif "invalid" in err or "api_key" in err:
                logger.error(f"Invalid API key detected: {e}")
                return None
            else:
                logger.warning(f"Attempt {attempt+1}: {e} — retrying in 3s")
                time.sleep(3)

    logger.warning("All 3 attempts failed — skipping sample")
    return None


def label_batch(samples: list[dict], output_path: Path, skip_existing: bool = True) -> list[dict]:
    """
    Label all samples with Gemini key rotation.
    - Saves after every sample (crash-safe — re-run to resume)
    - Shows live throughput and ETA
    - Logs cost estimate per key
    """
    existing = {}
    if skip_existing and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing[row["id"]] = row
        logger.info(f"Resuming — {len(existing)} already labelled, "
                    f"{len(samples) - len(existing)} remaining")

    labelled      = list(existing.values())
    cost_estimate = 0.0
    failed        = 0
    t_start       = time.time()

    pending = [s for s in samples if s["id"] not in existing]
    logger.info(f"Labelling {len(pending)} samples with {len(GEMINI_API_KEYS)} key(s) "
                f"@ {RPM_LIMIT * len(GEMINI_API_KEYS)} RPM effective throughput")

    with open(output_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(pending, desc="Labelling")):

            label = label_judgment(sample["text"])

            if label is None:
                failed += 1
                continue

            outcome              = (label.get("outcome") or "").lower().strip()
            label["outcome_label"] = OUTCOME_TO_LABEL.get(outcome, -1)

            row = {
                "id":     sample["id"],
                "source": sample["source"],
                "text":   sample["text"],
                "label":  label,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            labelled.append(row)

            tokens_used    = len(sample["text"][:4000]) // 4
            cost_estimate += (tokens_used / 1000) * 0.000075

            # Print ETA every 50 samples
            if (i + 1) % 50 == 0:
                elapsed  = time.time() - t_start
                rate     = (i + 1) / elapsed          # samples/sec
                remaining = len(pending) - (i + 1)
                eta_mins = (remaining / rate) / 60 if rate > 0 else 0
                logger.info(
                    f"Progress: {i+1}/{len(pending)} | "
                    f"Success: {len(labelled)-len(existing)} | "
                    f"Failed: {failed} | "
                    f"Cost so far: ${cost_estimate:.3f} | "
                    f"ETA: {eta_mins:.0f} min"
                )

    total_labelled = len(labelled) - len(existing)
    logger.success(
        f"Labelling complete:\n"
        f"  Newly labelled: {total_labelled}\n"
        f"  Failed/skipped: {failed}\n"
        f"  Total in file:  {len(labelled)}\n"
        f"  Estimated cost: ${cost_estimate:.3f}"
    )
    return labelled


# ── Step 3: Quality filtering ─────────────────────────────────────────────────

def is_valid_sample(row: dict) -> tuple[bool, str]:
    label = row.get("label", {})
    if not isinstance(label, dict):
        return False, "label_not_dict"

    for field in ["case_name", "petitioner", "respondent", "holding", "outcome", "court"]:
        val = label.get(field, "")
        if not val or str(val).strip() in ("", "null"):
            return False, f"missing_{field}"

    if len(row.get("text", "")) < 500:
        return False, "text_too_short"

    outcome = label.get("outcome", "").lower().strip()
    if outcome not in {"dismissed", "allowed", "disposed", "remanded", "modified", "withdrawn", "settled"}:
        return False, f"invalid_outcome_{outcome}"

    if not label.get("statutes_cited") and not label.get("legal_issues"):
        return False, "no_statutes_or_issues"

    if len(label.get("holding", "")) < 30:
        return False, "holding_too_short"

    return True, "ok"


def filter_dataset(labelled: list[dict]) -> tuple[list[dict], dict]:
    valid, invalid = [], {}
    for row in labelled:
        ok, reason = is_valid_sample(row)
        if ok:
            valid.append(row)
        else:
            invalid[reason] = invalid.get(reason, 0) + 1
    logger.info(f"Quality filter: {len(valid)} valid / {len(labelled)} total")
    logger.info(f"Rejection reasons: {invalid}")
    return valid, invalid


# ── Step 4: ChatML format ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Nyaya, a specialized Indian legal extraction model trained on Supreme Court "
    "and High Court judgments. Given a judgment text, extract all structured information "
    "and return it as a valid JSON object. Be precise with Indian legal citation formats "
    "(AIR, SCC, SCR). Never hallucinate statute sections or case citations not in the text."
)


def to_chatml(row: dict) -> dict:
    return {
        "id":     row["id"],
        "source": row["source"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Extract structured data from this Indian court "
                                             "judgment and return a JSON object:\n\n" + row["text"][:4000]},
            {"role": "assistant", "content": json.dumps(row["label"], indent=2, ensure_ascii=False)},
        ],
    }


# ── Step 5: Split and save ────────────────────────────────────────────────────

def split_and_save(valid: list[dict], test_size: int = 500, seed: int = 42) -> None:
    random.seed(seed)
    random.shuffle(valid)

    test_raw  = valid[:test_size]
    train_raw = valid[test_size:]

    logger.info(f"Split: {len(train_raw)} train / {len(test_raw)} test")

    with open(TEST_SET_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for row in test_raw:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.success(f"Test set saved: {len(test_raw)} samples")

    with open(PROCESSED_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_raw:
            f.write(json.dumps(to_chatml(row), ensure_ascii=False) + "\n")
    logger.success(f"Train set saved: {len(train_raw)} samples")

    stats = {
        "total": len(valid), "train": len(train_raw), "test": len(test_raw),
        "labeller": "gemini-2.5-flash (google-genai SDK)",
        "sources": {}, "outcomes": {},
    }
    for row in valid:
        src     = row.get("source", "unknown")
        outcome = row.get("label", {}).get("outcome", "unknown")
        stats["sources"][src]  = stats["sources"].get(src, 0) + 1
        stats["outcomes"][outcome] = stats["outcomes"].get(outcome, 0) + 1

    with open(PROCESSED_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.success(f"Stats: {stats['sources']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(max_samples: int = 10000):
    logger.info("=" * 60)
    logger.info("Nyaya-7B — Phase 1: Dataset Preparation")
    logger.info("=" * 60)

    raw_samples = collect_from_huggingface()
    raw_samples = deduplicate(raw_samples)

    if len(raw_samples) == 0:
        logger.error(
            "No samples collected. Before fixing the code, run:\n"
            "  python finetuning/inspect_datasets.py\n"
            "to see the actual column names of each dataset."
        )
        return

    # Interleave and sample using round-robin to ensure balanced dataset labeling
    raw_samples = round_robin_sample(raw_samples, max_samples)
    logger.info(f"Interleaved and sampled down to {len(raw_samples)} samples (max_samples={max_samples})")

    labelled_path = RAW_DIR / "labelled.jsonl"
    labelled      = label_batch(raw_samples, labelled_path)

    valid, _ = filter_dataset(labelled)
    if len(valid) < 500:
        logger.error(f"Only {len(valid)} valid samples — need at least 500.")
        return

    split_and_save(valid, test_size=min(500, len(valid) // 10))

    logger.info("=" * 60)
    logger.success("Phase 1 complete. Upload data/processed/train.jsonl to Kaggle.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()
    main(args.max_samples)
