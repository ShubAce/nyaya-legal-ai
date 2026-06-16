"""
evaluate_model.py — Post-training sanity checks on a freshly trained checkpoint.

Runs BEFORE the full benchmark (Phase 5). Catches obvious regressions early.
Checks: perplexity on test set, JSON validity rate, sample output quality.

Usage:
    python finetuning/evaluate_model.py --checkpoint checkpoints/nyaya-r16/final
"""

import json
import math
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

TEST_SET_PATH = Path("data/test_set/test.jsonl")
SAMPLE_LIMIT  = 100   # evaluate on first 100 test samples for speed


def load_test_samples(path: Path, limit: int = SAMPLE_LIMIT) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} test samples")
    return samples


def compute_perplexity(model, tokenizer, texts: list[str], max_len: int = 512) -> float:
    """Compute average perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            inputs = tokenizer(
                text[:2000],
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            ).to(model.device)

            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)

            # outputs.loss is mean NLL per token
            n_tokens = labels.shape[1]
            total_loss   += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_nll    = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    return round(perplexity, 2)


def check_json_validity(model, tokenizer, samples: list[dict]) -> dict:
    """
    Generate outputs for each sample and check JSON parse success rate.
    Returns stats dict.
    """
    valid, invalid, total = 0, 0, 0
    invalid_examples = []

    model.eval()
    for sample in tqdm(samples[:50], desc="Checking JSON validity"):
        text = sample["text"][:2000]
        prompt = (
            f"<s>[INST] Extract structured data from this Indian court judgment "
            f"and return a JSON object:\n\n{text} [/INST] "
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        total += 1
        try:
            json.loads(generated)
            valid += 1
        except json.JSONDecodeError:
            invalid += 1
            if len(invalid_examples) < 3:
                invalid_examples.append(generated[:300])

    validity_rate = valid / total if total > 0 else 0.0
    return {
        "json_valid":        valid,
        "json_invalid":      invalid,
        "total_checked":     total,
        "json_validity_rate": round(validity_rate, 3),
        "invalid_examples":  invalid_examples,
    }


def check_field_coverage(samples: list[dict], generated_outputs: list[dict]) -> dict:
    """
    Check what fraction of expected fields are present in generated JSON.
    """
    expected_fields = [
        "case_name", "court", "petitioner", "respondent",
        "statutes_cited", "legal_issues", "holding", "outcome"
    ]
    field_coverage = {f: 0 for f in expected_fields}
    total = len(generated_outputs)

    for output in generated_outputs:
        for field in expected_fields:
            val = output.get(field)
            if val and val not in ("", "null", None, [], {}):
                field_coverage[field] += 1

    return {
        field: round(count / total, 3) if total > 0 else 0.0
        for field, count in field_coverage.items()
    }


def run_evaluation(checkpoint_path: str) -> dict:
    checkpoint = Path(checkpoint_path)
    logger.info(f"Evaluating checkpoint: {checkpoint}")

    if not TEST_SET_PATH.exists():
        logger.error(f"Test set not found at {TEST_SET_PATH}. Run prepare_dataset.py first.")
        return {}

    # ── Load model and tokenizer ──────────────────────────────────────────────
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()

    # ── Load test samples ─────────────────────────────────────────────────────
    samples = load_test_samples(TEST_SET_PATH, limit=SAMPLE_LIMIT)

    # ── Perplexity ────────────────────────────────────────────────────────────
    texts = [s["text"] for s in samples]
    logger.info("Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, texts[:50])
    logger.info(f"Perplexity: {ppl}")

    # ── JSON validity ─────────────────────────────────────────────────────────
    logger.info("Checking JSON validity rate...")
    json_stats = check_json_validity(model, tokenizer, samples)
    logger.info(f"JSON validity: {json_stats['json_validity_rate']:.1%}")

    if json_stats["invalid_examples"]:
        logger.warning("Sample invalid outputs:")
        for ex in json_stats["invalid_examples"]:
            logger.warning(f"  {ex[:150]}...")

    # ── Summary ───────────────────────────────────────────────────────────────
    results = {
        "checkpoint":         str(checkpoint),
        "perplexity":         ppl,
        "json_validity_rate": json_stats["json_validity_rate"],
        "json_valid_count":   json_stats["json_valid"],
        "json_total":         json_stats["total_checked"],
    }

    # Pass/fail thresholds
    PASS_PERPLEXITY    = 15.0   # lower is better; base model is ~25-30 on legal text
    PASS_JSON_VALIDITY = 0.90   # 90% of outputs should be valid JSON

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Perplexity:    {ppl:6.2f}  {'✓ PASS' if ppl < PASS_PERPLEXITY else '✗ FAIL (too high)'}")
    logger.info(f"JSON Validity: {json_stats['json_validity_rate']:.1%}  "
                f"{'✓ PASS' if json_stats['json_validity_rate'] >= PASS_JSON_VALIDITY else '✗ FAIL (too low)'}")
    logger.info("=" * 50)

    # Save results
    results_path = checkpoint / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Nyaya-7B checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    args = parser.parse_args()
    run_evaluation(args.checkpoint)
