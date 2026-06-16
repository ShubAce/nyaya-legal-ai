"""
label_ollama.py — Local Ollama labeller for Nyaya-7B dataset preparation.

Replaces Gemini API labelling with a local model running on your GPU.
Designed for: RTX 4050 6GB + 16GB RAM + Intel i7

Recommended models (pick one based on your VRAM):
  mistral        — 7B, ~4.1GB VRAM in 4-bit  ← best quality/speed balance
  llama3.1       — 8B, ~4.7GB VRAM in 4-bit  ← slightly better instruction following
  phi3:medium    — 14B, ~8GB VRAM             ← needs more VRAM, higher quality
  gemma2:2b      — 2B, ~1.6GB VRAM            ← fast but lower quality

Setup (one-time):
    1. Download Ollama: https://ollama.com/download
    2. Pull a model: ollama pull mistral
    3. Confirm it runs: ollama run mistral "hello"

Usage:
    # Label 2000 samples (resumes from where labelled.jsonl left off)
    python finetuning/label_ollama.py --max-samples 2000

    # Use a different model
    python finetuning/label_ollama.py --model llama3.1 --max-samples 2000

    # After labelling, run this to split into train/test
    python finetuning/label_ollama.py --split-only

Expected speed on RTX 4050 6GB with mistral:
    ~8-12 seconds per sample → 2000 samples ≈ 5-7 hours
    Run overnight — it will finish by morning.
"""

import os
import sys
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

load_dotenv()

# ── Paths (same as prepare_dataset.py — shared labelled.jsonl) ───────────────
DATA_DIR      = Path("data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_SET_DIR  = DATA_DIR / "test_set"

for d in [RAW_DIR, PROCESSED_DIR, TEST_SET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LABELLED_PATH = RAW_DIR / "labelled.jsonl"   # shared with Gemini labeller

# ── Import schema ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from schema import SCHEMA_DESCRIPTION, OUTCOME_TO_LABEL

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Prompt ────────────────────────────────────────────────────────────────────
# Kept shorter than Gemini version — local models are slower so we minimise tokens
LABELLER_PROMPT = """You are a precise Indian legal data extraction system.
Extract structured information from the Indian court judgment below.
Return ONLY a valid JSON object. No explanation, no preamble, no markdown fences.

Schema:
{schema}

Judgment text:
{text}

JSON output:"""

SYSTEM_PROMPT_TRAIN = (
    "You are Nyaya, a specialized Indian legal extraction model trained on Supreme Court "
    "and High Court judgments. Given a judgment text, extract all structured information "
    "and return it as a valid JSON object. Be precise with Indian legal citation formats "
    "(AIR, SCC, SCR). Never hallucinate statute sections or case citations not in the text."
)


# ── Ollama health check ───────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    """Verify Ollama is running and the model is available."""
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        if r.status_code != 200:
            logger.error(f"Ollama not responding at {OLLAMA_URL}")
            return False

        available = [m["name"] for m in r.json().get("models", [])]
        # Model names can be "mistral" or "mistral:latest" — check both
        model_found = any(
            model in m or m.startswith(model)
            for m in available
        )

        if not model_found:
            logger.error(
                f"Model '{model}' not found in Ollama.\n"
                f"Available models: {available}\n"
                f"Pull it with: ollama pull {model}"
            )
            return False

        logger.success(f"Ollama running ✓  |  Model '{model}' available ✓")
        return True

    except httpx.ConnectError:
        logger.error(
            f"Cannot connect to Ollama at {OLLAMA_URL}\n"
            f"Make sure Ollama is running: open a terminal and run 'ollama serve'\n"
            f"Or download Ollama from: https://ollama.com/download"
        )
        return False


# ── Core labelling function ───────────────────────────────────────────────────

def label_judgment(text: str, model: str) -> Optional[dict]:
    """
    Call local Ollama model using /api/chat (chat format).
    Llama 3.1 is instruction-tuned — chat format gives much better
    structured JSON output than /api/generate.
    """
    truncated = text[:3000]
    user_msg  = (
        "Extract structured data from this Indian court judgment "
        "and return ONLY a valid JSON object matching the schema. "
        "No explanation, no preamble, no markdown fences.\n\n"
        f"Schema:\n{SCHEMA_DESCRIPTION}\n\n"
        f"Judgment text:\n{truncated}"
    )

    for attempt in range(3):
        try:
            response = httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":  model,
                    "stream": False,
                    "format": "json",       # forces JSON output
                    "options": {
                        "temperature":    0.0,
                        "num_predict":    800,
                        "num_ctx":        4096,
                        "repeat_penalty": 1.1,
                    },
                    "messages": [
                        {
                            "role":    "system",
                            "content": (
                                "You are a precise Indian legal data extraction system. "
                                "You extract structured information from court judgments "
                                "and return ONLY valid JSON. Never add explanation or markdown."
                            ),
                        },
                        {
                            "role":    "user",
                            "content": user_msg,
                        },
                    ],
                },
                timeout=120.0,
            )

            if response.status_code != 200:
                logger.warning(f"Attempt {attempt+1}: Ollama returned {response.status_code}")
                time.sleep(2)
                continue

            raw = response.json().get("message", {}).get("content", "").strip()
            if not raw:
                logger.warning(f"Attempt {attempt+1}: empty response")
                time.sleep(1)
                continue

            # Strip markdown fences defensively
            if raw.startswith("```"):
                parts = raw.split("```")
                raw   = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                logger.warning(f"Attempt {attempt+1}: got {type(parsed)}, expected dict")
                continue
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}: JSON decode failed — {str(e)[:80]}")
            if attempt == 2:
                return None
            time.sleep(1)

        except httpx.TimeoutException:
            logger.warning(f"Attempt {attempt+1}: timeout — model slow, retrying...")
            time.sleep(3)

        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


# ── Dataset loading (same as prepare_dataset.py) ─────────────────────────────

def collect_raw_samples(max_samples: int) -> list[dict]:
    """Load raw judgment texts from HuggingFace datasets."""
    logger.info("Loading HuggingFace datasets...")
    all_texts = []

    # ── IL-TUR/summ ──────────────────────────────────────────────────────────
    try:
        count_before = len(all_texts)
        ds = load_dataset("Exploration-Lab/IL-TUR", "summ",
                          split="train", trust_remote_code=False)
        for row in ds:
            doc = row.get("document", "")
            text = " ".join(doc) if isinstance(doc, list) else str(doc)
            if len(text) > 500:
                all_texts.append({
                    "text":   text,
                    "source": "IL-TUR-summ",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"IL-TUR/summ: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"IL-TUR/summ failed: {e}")

    # ── IL-TUR/bail ───────────────────────────────────────────────────────────
    try:
        count_before = len(all_texts)
        ds = load_dataset("Exploration-Lab/IL-TUR", "bail",
                          split="train_all", trust_remote_code=False)
        for row in ds:
            for col in ["text", "document", "judgment", "Text", "passage", "context"]:
                val = row.get(col, "")
                if isinstance(val, list):
                    val = " ".join(str(s) for s in val)
                if val and len(str(val)) > 500:
                    text = str(val)
                    all_texts.append({
                        "text":   text,
                        "source": "IL-TUR-bail",
                        "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                    })
                    break
        logger.success(f"IL-TUR/bail: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"IL-TUR/bail failed: {e}")

    # ── IL-TUR/cjpe ───────────────────────────────────────────────────────────
    try:
        count_before = len(all_texts)
        ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe",
                          split="single_train", trust_remote_code=False)
        for row in ds:
            for col in ["text", "document", "judgment", "Text", "facts", "context"]:
                val = row.get(col, "")
                if isinstance(val, list):
                    val = " ".join(str(s) for s in val)
                if val and len(str(val)) > 500:
                    all_texts.append({
                        "text":   str(val),
                        "source": "IL-TUR-cjpe",
                        "id":     hashlib.md5(str(val)[:100].encode()).hexdigest(),
                    })
                    break
        logger.success(f"IL-TUR/cjpe: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"IL-TUR/cjpe failed: {e}")

    # ── InJudgements (column = 'Text') ────────────────────────────────────────
    try:
        count_before = len(all_texts)
        ds = load_dataset("opennyaiorg/InJudgements_dataset",
                          split="train", trust_remote_code=False)
        for row in ds:
            text = row.get("Text", "")
            if text and len(text) > 500:
                all_texts.append({
                    "text":   text,
                    "source": "InJudgements",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"InJudgements: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"InJudgements failed: {e}")

    # ── RishiAI (column = 'Judgment') ─────────────────────────────────────────
    try:
        count_before = len(all_texts)
        ds = load_dataset("rishiai/indian-court-judgements-and-its-summaries",
                          split="train", trust_remote_code=False)
        for row in ds:
            text = row.get("Judgment", "")
            if text and len(text) > 500:
                all_texts.append({
                    "text":   text,
                    "source": "RishiAI",
                    "id":     hashlib.md5(text[:100].encode()).hexdigest(),
                })
        logger.success(f"RishiAI: {len(all_texts) - count_before} samples")
    except Exception as e:
        logger.warning(f"RishiAI failed: {e}")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    seen, unique = set(), []
    for s in all_texts:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)
    logger.info(f"After dedup: {len(unique)} unique samples")

    # ── Sample down ───────────────────────────────────────────────────────────
    if len(unique) > max_samples:
        random.seed(42)
        unique = random.sample(unique, max_samples)
        logger.info(f"Sampled down to {max_samples}")

    return unique


# ── Main labelling loop ───────────────────────────────────────────────────────

def label_all(samples: list[dict], model: str) -> list[dict]:
    """
    Label all samples with local Ollama model.
    Crash-safe: saves every sample to labelled.jsonl immediately.
    Resumable: skips samples already in labelled.jsonl on restart.
    """
    # Load existing labels — never re-label what's already done
    existing = {}
    if LABELLED_PATH.exists():
        with open(LABELLED_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    existing[row["id"]] = row
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resuming — {len(existing)} already labelled")

    pending = [s for s in samples if s["id"] not in existing]

    if not pending:
        logger.success("All samples already labelled — nothing to do!")
        return list(existing.values())

    logger.info(
        f"Labelling {len(pending)} samples with Ollama ({model})\n"
        f"  GPU:   RTX 4050 6GB\n"
        f"  Model: llama3.1:8b — 4.9GB VRAM, 128K context\n"
        f"  Speed: ~6-10 sec/sample (chat format, JSON mode)\n"
        f"  ETA:   {len(pending) * 8 / 3600:.1f} hours\n"
        f"  Tip:   Safe to Ctrl+C and re-run — resumes automatically from sample {len(existing)+1}"
    )

    labelled  = list(existing.values())
    failed    = 0
    t_start   = time.time()

    with open(LABELLED_PATH, "a", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(pending, desc=f"Ollama ({model})")):

            t_sample = time.time()
            label    = label_judgment(sample["text"], model)
            elapsed_sample = time.time() - t_sample

            if label is None:
                failed += 1
                logger.debug(f"Sample {i+1} failed — skipping")
                continue

            # Add outcome label integer
            outcome              = label.get("outcome", "").lower().strip()
            label["outcome_label"] = OUTCOME_TO_LABEL.get(outcome, -1)

            row = {
                "id":      sample["id"],
                "source":  sample["source"],
                "text":    sample["text"],
                "label":   label,
                "labeller": f"ollama/{model}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()   # ← write to disk immediately, crash-safe
            labelled.append(row)

            # Progress log every 25 samples
            if (i + 1) % 25 == 0:
                elapsed   = time.time() - t_start
                done      = i + 1
                rate      = done / elapsed
                remaining = len(pending) - done
                eta_h     = (remaining / rate) / 3600 if rate > 0 else 0
                eta_m     = (remaining / rate) / 60 if rate > 0 else 0
                logger.info(
                    f"  [{done}/{len(pending)}] "
                    f"success={len(labelled)-len(existing)} "
                    f"failed={failed} "
                    f"last={elapsed_sample:.1f}s "
                    f"ETA={eta_h:.1f}h ({eta_m:.0f}min)"
                )

    total_new = len(labelled) - len(existing)
    logger.success(
        f"\nLabelling complete:\n"
        f"  Newly labelled: {total_new}\n"
        f"  Failed/skipped: {failed}\n"
        f"  Total in file:  {len(labelled)}\n"
        f"  Saved to:       {LABELLED_PATH}"
    )
    return labelled


# ── Quality filtering ─────────────────────────────────────────────────────────

HINDI_OUTCOME_MAP = {
    "जमानत निरस्त":                                           "dismissed",
    "जमानत निरस्त नहीं की गई":                               "allowed",
    "जमानत निरस्त नहीं किया गया":                            "allowed",
    "निरस्त":                                                  "dismissed",
    "खारिज":                                                   "dismissed",
    "जमानत खारिज":                                            "dismissed",
    "जमानत प्रार्थनापत्र खारिज":                             "dismissed",
    "जमानत प्रार्थना पत्र खारिज":                            "dismissed",
    "जमानत प्रार्थनापत्र निरस्त नहीं किया गया":              "allowed",
    "जमानत प्रार्थना पत्र खारिज हुआ":                       "dismissed",
    "जमानत प्रार्थनापत्र स्वीकार किये जाने की <नाम> की गयी है": "allowed",
    "जमानत स्वीकार":                                          "allowed",
    "जमानत दी गई":                                            "allowed",
    "जमानत प्राप्त":                                          "allowed",
    "जमानत ना":                                               "dismissed",
    "जमानत रार्थना पत्र खारिज":                              "dismissed",
    "विचाराधीन":                                              "disposed",
    "conviction":                                              "dismissed",
    "acquitted":                                               "allowed",
    "reserved":                                               "disposed",
}

def is_valid_sample(row: dict) -> tuple[bool, str]:
    label = row.get("label", {})
    if not isinstance(label, dict):
        return False, "label_not_dict"

    for field in ["case_name", "petitioner", "respondent", "holding", "outcome", "court"]:
        val = label.get(field, "")
        if not val or str(val).strip() in ("", "null", "N/A", "n/a", "unknown"):
            return False, f"missing_{field}"

    if len(row.get("text", "")) < 500:
        return False, "text_too_short"

    outcome = label.get("outcome", "").lower().strip()

    # Normalize Hindi outcomes → English before checking
    hindi_normalized = HINDI_OUTCOME_MAP.get(label.get("outcome", "").strip())
    if hindi_normalized:
        label["outcome"] = hindi_normalized          # fix in place
        label["outcome_label"] = OUTCOME_TO_LABEL.get(hindi_normalized, -1)
        outcome = hindi_normalized

    if outcome not in {"dismissed", "allowed", "disposed", "remanded",
                       "modified", "withdrawn", "settled"}:
        return False, f"invalid_outcome_{outcome}"

    if not label.get("statutes_cited") and not label.get("legal_issues"):
        return False, "no_statutes_or_issues"

    if len(label.get("holding", "")) < 30:
        return False, "holding_too_short"

    return True, "ok"


# ── ChatML formatting ─────────────────────────────────────────────────────────

def to_chatml(row: dict) -> dict:
    return {
        "id":     row["id"],
        "source": row["source"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT_TRAIN},
            {"role": "user",      "content": "Extract structured data from this Indian court "
                                             "judgment and return a JSON object:\n\n"
                                             + row["text"][:4000]},
            {"role": "assistant", "content": json.dumps(row["label"], indent=2,
                                                        ensure_ascii=False)},
        ],
    }


# ── Split and save ────────────────────────────────────────────────────────────

def split_and_save(labelled: list[dict], test_size: int = 200) -> None:
    """Filter, split into train/test, and save as JSONL."""
    # Quality filter
    valid, rejected = [], {}
    for row in labelled:
        ok, reason = is_valid_sample(row)
        if ok:
            valid.append(row)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    logger.info(f"Quality filter: {len(valid)} valid / {len(labelled)} total")
    if rejected:
        logger.info(f"Rejected: {rejected}")

    if len(valid) < 200:
        logger.error(
            f"Only {len(valid)} valid samples — too few to train.\n"
            f"Try labelling more samples or loosening quality filters."
        )
        return

    # Split — keep test_size as held-out test set
    random.seed(42)
    random.shuffle(valid)
    actual_test = min(test_size, len(valid) // 10)
    test_raw    = valid[:actual_test]
    train_raw   = valid[actual_test:]

    # Save test set (raw rows — for evaluation)
    with open(TEST_SET_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for row in test_raw:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save train set (ChatML format — for finetuning)
    with open(PROCESSED_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_raw:
            f.write(json.dumps(to_chatml(row), ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "total":    len(valid),
        "train":    len(train_raw),
        "test":     len(actual_test if isinstance(actual_test, list) else test_raw),
        "labeller": f"ollama/{DEFAULT_MODEL}",
        "sources":  {},
        "outcomes": {},
    }
    for row in valid:
        src     = row.get("source", "unknown")
        outcome = row.get("label", {}).get("outcome", "unknown")
        stats["sources"][src]    = stats["sources"].get(src, 0) + 1
        stats["outcomes"][outcome] = stats["outcomes"].get(outcome, 0) + 1

    with open(PROCESSED_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.success(
        f"\nDataset ready:\n"
        f"  Train: {len(train_raw)} samples → data/processed/train.jsonl\n"
        f"  Test:  {len(test_raw)} samples  → data/test_set/test.jsonl\n"
        f"  Sources: {stats['sources']}\n"
        f"\nNext step: upload data/processed/train.jsonl to Kaggle dataset"
    )


# ── Split-only mode — run after labelling is done ─────────────────────────────

def split_only() -> None:
    """Load existing labelled.jsonl and just do the split. No new labelling."""
    if not LABELLED_PATH.exists():
        logger.error(f"No labelled data found at {LABELLED_PATH}")
        logger.error("Run labelling first: python finetuning/label_ollama.py --max-samples 2000")
        return

    labelled = []
    with open(LABELLED_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                labelled.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(labelled)} labelled samples from {LABELLED_PATH}")
    split_and_save(labelled)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label Indian legal judgments with local Ollama model"
    )
    parser.add_argument(
        "--model",        type=str, default=DEFAULT_MODEL,
        help="Ollama model name (default: mistral). Try: llama3.1, phi3:medium, gemma2:2b"
    )
    parser.add_argument(
        "--max-samples",  type=int, default=2000,
        help="Max samples to collect and label (default: 2000)"
    )
    parser.add_argument(
        "--split-only",   action="store_true",
        help="Skip labelling — just split existing labelled.jsonl into train/test"
    )
    parser.add_argument(
        "--ollama-url",   type=str, default=OLLAMA_URL,
        help=f"Ollama server URL (default: {OLLAMA_URL})"
    )
    args = parser.parse_args()

    # Split-only mode — no Ollama needed
    if args.split_only:
        split_only()
        return

    logger.info("=" * 60)
    logger.info("Nyaya-7B — Local Ollama Labeller")
    logger.info("=" * 60)
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Ollama URL:  {args.ollama_url}")
    logger.info(f"  Resume from: {LABELLED_PATH}")
    logger.info("=" * 60)

    # Verify Ollama is running
    if not check_ollama(args.model):
        sys.exit(1)

    # Collect raw samples (fast — just downloads datasets)
    raw_samples = collect_raw_samples(args.max_samples)
    if not raw_samples:
        logger.error("No samples collected. Check dataset availability.")
        sys.exit(1)

    # Label with Ollama (slow — runs local model)
    labelled = label_all(raw_samples, args.model)

    # Filter, split, save
    if len(labelled) >= 200:
        split_and_save(labelled)
    else:
        logger.warning(
            f"Only {len(labelled)} samples labelled so far.\n"
            f"Re-run to continue: python finetuning/label_ollama.py --max-samples {args.max_samples}"
        )


if __name__ == "__main__":
    main()




