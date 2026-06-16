"""
extraction_agent.py — Agent 1: Calls the finetuned Nyaya-7B model to extract
structured JSON from raw judgment text.

This is the core agent that makes finetuning necessary. A base model or
prompted Gemini 1.5 Pro will hallucinate Indian-specific citation formats and
misparse legal English. Nyaya-7B, trained on 10K judgments, handles these
natively.
"""

import os
import json
import re
import time
from pathlib import Path
from functools import lru_cache
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from loguru import logger
from dotenv import load_dotenv

from agents.state import LegalExtractionState

load_dotenv()

MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "kaggle/nyaya7b-finetuning-output/nyaya-checkpoints/nyaya-7b-merged")
_BACKEND_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _load_pipeline():
    """
    Load the Nyaya-7B model once and cache it for the process lifetime.
    Uses lru_cache so it's only loaded on first call.
    Loads in 4-bit format via bitsandbytes if GPU is available, else float32 on CPU.
    """
    model_source = MODEL_PATH
    model_path = Path(model_source)
    if not model_path.is_absolute():
        model_path = _BACKEND_DIR / model_path

    if not model_path.exists():
        fallback_path = _BACKEND_DIR / "kaggle" / "nyaya7b-finetuning-output" / "nyaya-checkpoints" / "nyaya-7b-merged"
        if fallback_path.exists():
            model_path = fallback_path
        else:
            raise RuntimeError(
                f"Local model not found. Checked:\n  {model_path}\n  {fallback_path}\n"
                f"Set FINETUNED_MODEL_PATH in .env to the correct path."
            )

    use_gpu = torch.cuda.is_available()
    logger.info(f"Loading Nyaya-7B from: {model_path} ({'GPU 4-bit' if use_gpu else 'CPU fp32'})")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token

    if use_gpu:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            logger.success("Using GPU with 4-bit quantization")
        except Exception as e:
            logger.warning(f"4-bit GPU load failed ({e}), falling back to CPU float32")
            use_gpu = False

    if not use_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.warning("Running on CPU — inference will be slow (~5-15 min per judgment)")

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if use_gpu else None,
    )

    elapsed = time.time() - start
    logger.success(f"Nyaya-7B loaded in {elapsed:.1f}s")
    return pipe


EXTRACTION_SYSTEM = (
    "You are Nyaya, a specialized Indian legal extraction model. "
    "Given a judgment text, extract all structured information and return "
    "a single valid JSON object. Never hallucinate statute sections or citations "
    "not present in the text. Return only JSON — no preamble, no explanation."
)


def _build_prompt(judgment_text: str, tokenizer) -> str:
    """Build prompt using tokenizer's chat template to match training format exactly."""
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"Extract structured data from this Indian court judgment and return a JSON object:\n\n{judgment_text[:3000]}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_json(raw_output: str) -> Optional[dict]:
    """
    Robustly extract JSON from model output.
    Handles: clean JSON, JSON with markdown fences, JSON embedded in text.
    """
    raw = raw_output.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { ... } block
    brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 4: fix common truncation — add closing brace
    if raw.startswith("{") and not raw.endswith("}"):
        try:
            return json.loads(raw + "}")
        except json.JSONDecodeError:
            pass

    return None


def extraction_agent(state: LegalExtractionState) -> LegalExtractionState:
    """
    Agent 1: Extract structured data from judgment text using Nyaya-7B.

    Writes to state:
        - raw_extraction: dict or None
        - audit_trail: appends extraction event
        - errors: appends any extraction errors
    """
    audit = state.get("audit_trail", [])
    errors = state.get("errors", [])

    judgment_text = state["judgment_text"]
    if not judgment_text or len(judgment_text.strip()) < 100:
        errors.append("extraction_agent: judgment_text too short or empty")
        return {**state, "raw_extraction": None, "audit_trail": audit, "errors": errors}

    try:
        pipe = _load_pipeline()
        prompt = _build_prompt(judgment_text, pipe.tokenizer)

        t0 = time.time()
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

        raw_text = outputs[0]["generated_text"]
        extracted = _extract_json(raw_text)

        if extracted is None:
            errors.append(f"extraction_agent: could not parse JSON from output: {raw_text[:200]}")
            audit.append({
                "agent":   "extraction",
                "status":  "json_parse_failed",
                "latency": round(elapsed, 2),
            })
            return {**state, "raw_extraction": None, "audit_trail": audit, "errors": errors}

        fields_found = [k for k, v in extracted.items() if v not in (None, "", [], {})]
        audit.append({
            "agent":       "extraction",
            "status":      "success",
            "fields_found": fields_found,
            "field_count":  len(fields_found),
            "latency":     round(elapsed, 2),
        })
        logger.info(f"Extraction complete: {len(fields_found)} fields in {elapsed:.1f}s")

    except Exception as e:
        error_msg = f"extraction_agent: unexpected error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        audit.append({"agent": "extraction", "status": "error", "error": str(e)})
        return {**state, "raw_extraction": None, "audit_trail": audit, "errors": errors}

    return {
        **state,
        "raw_extraction": extracted,
        "audit_trail":    audit,
        "errors":         errors,
    }
