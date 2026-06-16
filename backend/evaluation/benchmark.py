"""
benchmark.py — Phase 5: Run all four systems on the held-out test set
and produce the headline benchmark table.

Systems compared:
  1. Mistral-7B base  (no finetuning, prompted)
  2. Nyaya-7B         (your finetuned model)        ← should win
  3. Gemini 1.5 Flash (prompted, API)               ← cheap baseline
  4. Gemini 1.5 Pro   (prompted, API)               ← the bar to beat

Usage:
    python evaluation/benchmark.py --systems all --limit 500
    python evaluation/benchmark.py --systems nyaya gemini_pro --limit 50
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import datasets first to resolve Windows OpenMP/CUDA DLL collision between PyArrow and PyTorch
import datasets

import os
import json
import time
import argparse
import re
import hashlib
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from google import genai
from google.genai import types
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
import torch

from evaluation.metrics import compute_all_metrics, aggregate_metrics

load_dotenv()

# ── Gemini setup ──────────────────────────────────────────────────────────────
class GeminiKeyManager:
    def __init__(self):
        self.keys = []
        # Try keys GEMINI_API_KEY_1 to GEMINI_API_KEY_9 first
        for i in range(1, 10):
            k = os.getenv(f"GEMINI_API_KEY_{i}")
            if k:
                self.keys.append(k)
        
        # If no numbered keys are found, fall back to GEMINI_API_KEY
        if not self.keys:
            fb = os.getenv("GEMINI_API_KEY")
            if fb:
                self.keys.append(fb)
                
        if not self.keys:
            raise RuntimeError(
                "No Gemini API keys found in environment variables (GEMINI_API_KEY_1 to GEMINI_API_KEY_9, or GEMINI_API_KEY)"
            )
        
        self.current_index = 0
        self.client = genai.Client(api_key=self.keys[self.current_index])
        logger.info(f"GeminiKeyManager initialized with {len(self.keys)} keys. Current key index: {self.current_index}")

    def get_client(self) -> genai.Client:
        return self.client

    def rotate_key(self):
        if len(self.keys) <= 1:
            logger.warning("Only 1 Gemini API key available. Cannot rotate.")
            return self.client
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        key_preview = self.keys[self.current_index][:12] + "..." if len(self.keys[self.current_index]) > 12 else self.keys[self.current_index]
        logger.warning(
            f"Quota reached or rate limit hit. Rotating Gemini API key to index {self.current_index} (starts with {key_preview})"
        )
        self.client = genai.Client(api_key=self.keys[self.current_index])
        return self.client


# Instantiate a global/shared key manager
gemini_key_manager = None

def get_gemini_key_manager() -> GeminiKeyManager:
    global gemini_key_manager
    if gemini_key_manager is None:
        gemini_key_manager = GeminiKeyManager()
    return gemini_key_manager

TEST_SET_PATH = Path("data/test_set/test.jsonl")
RESULTS_DIR   = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Extraction prompt (shared across all systems) ─────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a precise Indian legal data extraction system.
Extract structured information from the judgment text and return ONLY a valid JSON object.
No preamble, no explanation, no markdown fences. Just raw JSON.

Required fields:
{
  "case_name": "<Petitioner v. Respondent>",
  "citation": "<AIR/SCC citation or null>",
  "court": "<full court name>",
  "year": <integer year or null>,
  "petitioner": "<petitioner name>",
  "respondent": "<respondent name>",
  "subject_matter": "<Criminal|Civil|Constitutional|...>",
  "statutes_cited": [{"act": "<act name>", "section": "<number>", "description": "<brief>"}],
  "precedents_cited": [{"citation": "<AIR/SCC>", "case_name": "<name or null>"}],
  "legal_issues": ["<issue 1>", "<issue 2>"],
  "holding": "<court's decision and reasoning in 1-3 sentences>",
  "outcome": "<dismissed|allowed|disposed|remanded|modified>"
}"""


def _parse_output(raw: str) -> dict:
    """Robustly parse JSON from model output, stripping any markdown fences."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


# ── Base class ────────────────────────────────────────────────────────────────

class BaseSystem(ABC):
    name: str
    cost_per_1k_input_tokens: float = 0.0

    @abstractmethod
    def predict(self, judgment_text: str) -> tuple[str, dict]:
        """Returns (raw_output_string, parsed_dict)."""
        pass

    def last_cost(self, input_tokens: int) -> float:
        return (input_tokens / 1000) * self.cost_per_1k_input_tokens


# ── System 1: Mistral-7B base (no finetuning) ────────────────────────────────

class MistralBaseSystem(BaseSystem):
    name = "mistral_base"
    cost_per_1k_input_tokens = 0.0

    def __init__(self):
        model_id  = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
        logger.info(f"Loading Mistral base in 4-bit: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def predict(self, judgment_text: str) -> tuple[str, dict]:
        prompt = (
            f"<s>[INST] {EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"Extract from this judgment:\n\n{judgment_text[:3000]} [/INST] "
        )
        raw = self.pipe(
            prompt, max_new_tokens=512, do_sample=False,
            return_full_text=False
        )[0]["generated_text"]
        return raw, _parse_output(raw)


# ── System 2: Nyaya-7B (your finetuned model) ────────────────────────────────

class NyayaFinetunedSystem(BaseSystem):
    name = "nyaya_7b"
    cost_per_1k_input_tokens = 0.0

    def __init__(self):
        from agents.extraction_agent import _load_pipeline
        self.pipe = _load_pipeline()

    def predict(self, judgment_text: str) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": "You are Nyaya, a specialized Indian legal extraction model."},
            {"role": "user", "content": f"Extract structured data from this judgment and return JSON:\n\n{judgment_text[:3000]}"}
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw = self.pipe(
            prompt, max_new_tokens=512, do_sample=False,
            return_full_text=False,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]
        return raw, _parse_output(raw)


# ── System 3: Gemini 1.5 Flash (cheap, fast baseline) ────────────────────────

class GeminiFlashSystem(BaseSystem):
    name = "gemini_flash"
    cost_per_1k_input_tokens = 0.000075   # $0.000075/1K input tokens

    def __init__(self):
        self.manager = get_gemini_key_manager()

    def predict(self, judgment_text: str) -> tuple[str, dict]:
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nExtract from:\n\n{judgment_text[:6000]}"
        max_attempts = len(self.manager.keys) * 2
        
        for attempt in range(max_attempts):
            client = self.manager.get_client()
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw = response.text.strip()
                return raw, _parse_output(raw)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = (
                    "429" in err_msg or 
                    "quota" in err_msg or 
                    "rate limit" in err_msg or 
                    "exhausted" in err_msg or 
                    "resource_exhausted" in err_msg
                )
                if is_rate_limit:
                    logger.warning(
                        f"Gemini Flash API rate limit / quota exceeded (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    self.manager.rotate_key()
                    time.sleep(1)
                else:
                    logger.warning(f"Gemini Flash call failed with non-rate-limit error: {e}")
                    time.sleep(2)
                    return "", {}
                    
        logger.error("All Gemini API keys exhausted or rate limited for Gemini Flash.")
        return "", {}


# ── System 4: Gemini 1.5 Pro (the bar to beat) ───────────────────────────────

class GeminiProSystem(BaseSystem):
    name = "gemini_pro"
    cost_per_1k_input_tokens = 0.00125    # $0.00125/1K input tokens

    def __init__(self):
        self.manager = get_gemini_key_manager()

    def predict(self, judgment_text: str) -> tuple[str, dict]:
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nExtract from:\n\n{judgment_text[:8000]}"
        max_attempts = len(self.manager.keys) * 2
        
        for attempt in range(max_attempts):
            client = self.manager.get_client()
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw = response.text.strip()
                return raw, _parse_output(raw)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = (
                    "429" in err_msg or 
                    "quota" in err_msg or 
                    "rate limit" in err_msg or 
                    "exhausted" in err_msg or 
                    "resource_exhausted" in err_msg
                )
                if is_rate_limit:
                    logger.warning(
                        f"Gemini Pro API rate limit / quota exceeded (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    self.manager.rotate_key()
                    time.sleep(1)
                else:
                    logger.warning(f"Gemini Pro call failed with non-rate-limit error: {e}")
                    time.sleep(2)
                    return "", {}
                    
        logger.error("All Gemini API keys exhausted or rate limited for Gemini Pro.")
        return "", {}


# ── System map ────────────────────────────────────────────────────────────────

SYSTEM_MAP = {
    "mistral_base": MistralBaseSystem,
    "nyaya":        NyayaFinetunedSystem,
    "gemini_flash": GeminiFlashSystem,
    "gemini_pro":   GeminiProSystem,
}

# Colors for charts
SYSTEM_COLORS = {
    "mistral_base": "#94a3b8",
    "nyaya_7b":     "#3b82f6",
    "gemini_flash": "#f59e0b",
    "gemini_pro":   "#10b981",
}


# ── Benchmark runner ──────────────────────────────────────────────────────────

def load_test_set(limit: Optional[int] = None) -> list[dict]:
    samples = []
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        for line in f:
            if limit and len(samples) >= limit:
                break
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} test samples")
    return samples


def get_text_hash(text: str) -> str:
    """Helper to generate a stable hash for judgment text to identify samples."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_system_benchmark(system: BaseSystem, samples: list[dict], run_id: str) -> dict:
    checkpoint_file = RESULTS_DIR / f"checkpoint_{system.name}_{run_id}.jsonl"
    
    # Load existing progress
    completed_runs = {}
    if checkpoint_file.exists():
        logger.info(f"Found existing checkpoint file for {system.name}: {checkpoint_file}. Loading progress...")
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        completed_runs[data["text_hash"]] = {
                            "metrics": data["metrics"],
                            "is_error": data.get("is_error", False)
                        }
            logger.info(f"Loaded {len(completed_runs)} completed samples from checkpoint.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}. Starting from scratch.")
            completed_runs = {}

    per_sample = []
    errors = 0

    # Populate per_sample with already completed metrics
    for sample in samples:
        judgment_text = sample.get("text", "")
        text_hash = get_text_hash(judgment_text)
        if text_hash in completed_runs:
            c = completed_runs[text_hash]
            per_sample.append(c["metrics"])
            if c["is_error"]:
                errors += 1

    pbar = tqdm(samples, desc=f"Benchmarking {system.name}")
    for sample in pbar:
        judgment_text = sample.get("text", "")
        text_hash = get_text_hash(judgment_text)
        
        if text_hash in completed_runs:
            continue

        t0 = time.time()
        is_err = False
        try:
            raw_output, predicted = system.predict(judgment_text)
        except KeyboardInterrupt:
            logger.info("\nBenchmark interrupted by user (Ctrl+C). Saving current progress...")
            pbar.close()
            raise
        except Exception as e:
            logger.warning(f"{system.name} prediction failed: {e}")
            raw_output, predicted = "", {}
            errors += 1
            is_err = True

        elapsed    = time.time() - t0
        input_toks = len(judgment_text[:6000]) // 4
        cost       = system.last_cost(input_toks)

        metrics              = compute_all_metrics(raw_output, predicted, sample.get("label", {}), cost)
        metrics["latency_s"] = round(elapsed, 2)
        
        per_sample.append(metrics)
        
        # Save progress to checkpoint file immediately
        checkpoint_data = {
            "text_hash": text_hash,
            "metrics": metrics,
            "is_error": is_err
        }
        try:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(checkpoint_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write checkpoint to {checkpoint_file}: {e}")

    # Check if we have any samples in per_sample
    if not per_sample:
        logger.warning(f"No samples were evaluated for {system.name}.")
        return {
            "system": system.name,
            "error_count": errors,
            "statute_f1": 0.0,
            "outcome_correct": 0.0,
            "party_accuracy": 0.0,
            "json_valid": 0.0,
            "hallucination_rate": 0.0,
            "field_coverage": 0.0,
            "cost_usd": 0.0
        }

    aggregated               = aggregate_metrics(per_sample)
    aggregated["system"]     = system.name
    aggregated["error_count"] = errors
    return aggregated


def print_results_table(results: list[dict]):
    df = pd.DataFrame(results).set_index("system")
    display_cols = [
        "statute_f1", "outcome_correct", "party_accuracy",
        "json_valid", "hallucination_rate", "field_coverage", "cost_usd",
    ]
    display_df = df[display_cols].round(3)
    display_df.columns = [
        "Statute F1", "Outcome Acc", "Party Acc",
        "JSON Valid", "Halluc. Rate", "Field Coverage", "Cost/judgment",
    ]

    print("\n" + "=" * 80)
    print("NYAYA-7B BENCHMARK RESULTS")
    print("=" * 80)
    print(display_df.to_string())
    print("=" * 80)

    # Headline: Nyaya vs Gemini Pro
    if "nyaya_7b" in df.index and "gemini_pro" in df.index:
        n_f1  = df.loc["nyaya_7b",   "statute_f1"]
        g_f1  = df.loc["gemini_pro", "statute_f1"]
        n_hr  = df.loc["nyaya_7b",   "hallucination_rate"]
        g_hr  = df.loc["gemini_pro", "hallucination_rate"]
        n_cost = df.loc["nyaya_7b",  "cost_usd"]
        g_cost = df.loc["gemini_pro","cost_usd"]

        print(f"\nHeadline results (Nyaya-7B vs Gemini 1.5 Pro):")
        print(f"  Statute F1:         {n_f1:.3f} vs {g_f1:.3f}  "
              f"({'WIN' if n_f1 > g_f1 else 'LOSS'} by {abs(n_f1-g_f1):.3f})")
        print(f"  Hallucination rate: {n_hr:.1%} vs {g_hr:.1%}  "
              f"({'WIN' if n_hr < g_hr else 'LOSS'})")
        print(f"  Cost per judgment:  ${n_cost:.4f} vs ${g_cost:.4f}  "
              f"(Nyaya saves ${g_cost-n_cost:.4f}/judgment)")


def save_results(results: list[dict], run_id: str):
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"benchmark_{ts}.json"
    csv_path  = RESULTS_DIR / f"benchmark_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "timestamp": ts, "results": results}, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.success(f"Results saved:\n  {json_path}\n  {csv_path}")
    return json_path, csv_path


def plot_benchmark(results: list[dict], save_path: Path):
    metrics = ["statute_f1", "outcome_correct", "party_accuracy", "json_valid", "field_coverage"]
    labels  = ["Statute F1", "Outcome Acc", "Party Acc", "JSON Valid", "Field Coverage"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart — all metrics
    ax = axes[0]
    df = pd.DataFrame(results).set_index("system")
    colors = [SYSTEM_COLORS.get(s, "#aaa") for s in df.index]
    df[metrics].plot(kind="bar", ax=ax)
    ax.set_title("Benchmark Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.legend(labels, loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Hallucination rate — lower is better
    ax2      = axes[1]
    systems  = [r["system"] for r in results]
    hallucs  = [r["hallucination_rate"] for r in results]
    bcolors  = [SYSTEM_COLORS.get(s, "#aaa") for s in systems]
    bars     = ax2.bar(systems, hallucs, color=bcolors)
    ax2.set_title("Hallucination Rate (lower = better)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Hallucination Rate")
    ax2.set_xticklabels(systems, rotation=20, ha="right")
    ax2.set_ylim(0, max(hallucs) * 1.3 if hallucs else 1)
    for bar, val in zip(bars, hallucs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"Benchmark chart saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Nyaya-7B benchmark runner")
    parser.add_argument("--systems", nargs="+", default=["nyaya"],
                        choices=list(SYSTEM_MAP.keys()) + ["all"])
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--run-id",  type=str, default="benchmark")
    args = parser.parse_args()

    system_names = list(SYSTEM_MAP.keys()) if "all" in args.systems else args.systems

    logger.info(f"Benchmarking: {system_names}")
    samples     = load_test_set(limit=args.limit)
    all_results = []

    try:
        for name in system_names:
            logger.info(f"\n{'='*40}\nRunning: {name}\n{'='*40}")
            system = SYSTEM_MAP[name]()
            result = run_system_benchmark(system, samples, args.run_id)
            all_results.append(result)

            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        logger.info("\nBenchmark execution paused by user (Ctrl+C). You can resume by running the same command again.")
        sys.exit(0)

    print_results_table(all_results)
    save_results(all_results, args.run_id)
    plot_benchmark(all_results, RESULTS_DIR / f"benchmark_{args.run_id}.png")

    # Clean up checkpoint files on successful completion of all systems
    for name in system_names:
        checkpoint_file = RESULTS_DIR / f"checkpoint_{name}_{args.run_id}.jsonl"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info(f"Removed temporary checkpoint file: {checkpoint_file}")
            except Exception as e:
                pass

    logger.success("Benchmark complete.")


if __name__ == "__main__":
    main()
