"""
extraction_agent.py — Agent 1: Calls the finetuned Nyaya-7B model to extract
structured JSON from raw judgment text.

This is the core agent that makes finetuning necessary. A base model or
prompted Gemini 1.5 Pro will hallucinate Indian-specific citation formats and
misparse legal English. Nyaya-7B, trained on 10K judgments, handles these
natively.

The model is loaded from the local merged checkpoint only (never the Hugging
Face Hub) in 4-bit on the GPU, which needs roughly 4.5 GB of VRAM.
"""

import json
import re
import threading
import time
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from loguru import logger

from agents.config import MAX_NEW_TOKENS, MODEL_PATH
from agents.state import LegalExtractionState

# ── Prompt — must match finetuning/prepare_dataset.py + the Kaggle training notebook ──

SYSTEM_PROMPT = (
    "You are Nyaya, a specialized Indian legal extraction model trained on Supreme Court "
    "and High Court judgments. Given a judgment text, extract all structured information "
    "and return it as a valid JSON object. Be precise with Indian legal citation formats "
    "(AIR, SCC, SCR). Never hallucinate statute sections or case citations not in the text."
)
USER_PREFIX     = "Extract structured data from this Indian court judgment and return a JSON object:\n\n"
MAX_INPUT_CHARS = 4000   # training samples were truncated to the first 4000 characters


def build_prompt(judgment_text: str) -> str:
    """
    Reproduce the exact training string (minus the answer). Training used
    f"<s>[INST] {system}\\n\\n{user} [/INST] {answer}</s>", not the tokenizer's
    chat template, which drops the space before [/INST].
    """
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{USER_PREFIX}{judgment_text.strip()[:MAX_INPUT_CHARS]} [/INST]"


# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_load_error: Optional[str] = None
_load_lock = threading.Lock()
_generate_lock = threading.Lock()   # one generation at a time on the GPU


def is_model_loaded() -> bool:
    return _model is not None


def model_load_error() -> Optional[str]:
    return _load_error


def load_model():
    """Load Nyaya-7B once (thread-safe) and return (model, tokenizer)."""
    global _model, _tokenizer, _load_error

    with _load_lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            _model, _tokenizer = _load_from_disk()
            _load_error = None
        except Exception as e:
            _load_error = str(e)
            raise
        return _model, _tokenizer


def _load_from_disk():
    if not (MODEL_PATH / "config.json").exists():
        raise RuntimeError(
            f"Nyaya-7B not found at {MODEL_PATH}. Set FINETUNED_MODEL_PATH in backend/.env "
            f"to the folder containing config.json and model.safetensors."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nyaya-7B needs a CUDA GPU (about 4.5 GB VRAM in 4-bit); torch.cuda.is_available() "
            "is False. Install a CUDA build of PyTorch. CPU inference would need ~30 GB of RAM."
        )

    logger.info(f"Loading Nyaya-7B (4-bit, GPU) from {MODEL_PATH}")
    start = time.time()

    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    # The checkpoint was saved by transformers 5.x, which stores rope_theta inside
    # `rope_parameters`. transformers 4.x ignores that key and silently falls back to
    # rope_theta=10000 instead of Mistral v0.3's 1e6, which corrupts every output.
    rope_theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    if rope_theta and getattr(config, "rope_theta", None) != rope_theta:
        config.rope_theta = rope_theta

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    logger.success(f"Nyaya-7B loaded in {time.time() - start:.1f}s ({vram:.1f} GB VRAM)")
    return model, tokenizer


def generate_extraction(judgment_text: str) -> str:
    """Run Nyaya-7B on a judgment and return the raw generated text."""
    model, tokenizer = load_model()
    # The prompt already starts with <s> and the tokenizer adds another BOS — this
    # matches how TRL tokenised the training strings, so it is kept on purpose.
    inputs = tokenizer(build_prompt(judgment_text), return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    with _generate_lock, torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── JSON recovery ─────────────────────────────────────────────────────────────

def _extract_json(raw_output: str) -> Optional[dict]:
    """
    Pull the JSON object out of the model output. Handles markdown fences,
    trailing text after the object, and output cut off at max_new_tokens.
    """
    start = raw_output.find("{")
    if start == -1:
        return None
    text = raw_output[start:]

    stack: list[str] = []
    in_string = escaped = False
    cut, cut_stack = None, None   # last point where the JSON prefix is complete

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                try:
                    return json.loads(text[:i + 1])
                except json.JSONDecodeError:
                    return None
            cut, cut_stack = i + 1, list(stack)
        elif ch == ",":
            cut, cut_stack = i, list(stack)

    # Truncated output: drop the incomplete tail and close the open brackets.
    if cut is None:
        return None
    try:
        return json.loads(text[:cut] + "".join(reversed(cut_stack)))
    except json.JSONDecodeError:
        return None


# ── Output normalisation ──────────────────────────────────────────────────────

def _as_list(value) -> list:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _as_text(value) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value))


def _as_statute(item) -> Optional[dict]:
    if isinstance(item, dict):
        return {
            "act":         _as_text(item.get("act")),
            "section":     _as_text(item.get("section")),
            "description": _as_text(item.get("description")),
        }
    if isinstance(item, str) and item.strip():
        # e.g. "Section 302 IPC" — the statute agent parses the act from this.
        return {"act": item.strip(), "section": item.strip(), "description": ""}
    return None


def _as_precedent(item) -> Optional[dict]:
    if isinstance(item, dict):
        citation = _as_text(item.get("citation"))
        case_name = _as_text(item.get("case_name"))
        return {"citation": citation, "case_name": case_name} if citation or case_name else None
    if isinstance(item, str) and item.strip():
        return {"citation": item.strip(), "case_name": ""}
    return None


def _as_year(value) -> Optional[int]:
    match = re.search(r"\b(1[89]\d\d|20\d\d)\b", str(value or ""))
    return int(match.group(1)) if match else None


def normalize_extraction(extracted: dict) -> dict:
    """Coerce the model's JSON into the schema the downstream agents expect."""
    out = dict(extracted)
    for key in ("case_name", "citation", "court", "petitioner", "respondent",
                "subject_matter", "holding"):
        out[key] = _as_text(extracted.get(key))
    out["outcome"]          = _as_text(extracted.get("outcome")).lower()
    out["year"]             = _as_year(extracted.get("year"))
    out["bench"]            = [_as_text(b) for b in _as_list(extracted.get("bench")) if _as_text(b)]
    out["legal_issues"]     = [_as_text(i) for i in _as_list(extracted.get("legal_issues")) if _as_text(i)]
    out["statutes_cited"]   = [s for s in map(_as_statute, _as_list(extracted.get("statutes_cited"))) if s]
    out["precedents_cited"] = [p for p in map(_as_precedent, _as_list(extracted.get("precedents_cited"))) if p]
    return out


# ── Agent ─────────────────────────────────────────────────────────────────────

def extraction_agent(state: LegalExtractionState) -> dict:
    """
    Agent 1: Extract structured data from judgment text using Nyaya-7B.

    Returns:
        - raw_extraction: normalised dict, or None on failure
        - audit_trail / errors: new entries for this agent
    """
    judgment_text = state.get("judgment_text") or ""
    if len(judgment_text.strip()) < 100:
        return {
            "raw_extraction": None,
            "audit_trail": [{"agent": "extraction", "status": "skipped", "reason": "text_too_short"}],
            "errors": ["extraction_agent: judgment_text too short or empty"],
        }

    t0 = time.time()
    try:
        raw_text = generate_extraction(judgment_text)
    except Exception as e:
        logger.error(f"extraction_agent: {e}")
        return {
            "raw_extraction": None,
            "audit_trail": [{"agent": "extraction", "status": "error", "error": str(e)}],
            "errors": [f"extraction_agent: {e}"],
        }
    elapsed = round(time.time() - t0, 2)

    extracted = _extract_json(raw_text)
    if not isinstance(extracted, dict):
        logger.warning(f"extraction_agent: JSON parse failed. Raw output ({len(raw_text)} chars):\n{raw_text[:400]}")
        return {
            "raw_extraction": None,
            "audit_trail": [{"agent": "extraction", "status": "json_parse_failed",
                             "latency": elapsed, "raw_text": raw_text[:500]}],
            "errors": [f"extraction_agent: could not parse JSON from model output: {raw_text[:200]}"],
        }

    extracted = normalize_extraction(extracted)
    fields_found = [k for k, v in extracted.items() if v not in (None, "", [], {})]
    logger.info(f"Extraction complete: {len(fields_found)} fields in {elapsed:.1f}s")

    return {
        "raw_extraction": extracted,
        "audit_trail": [{
            "agent":        "extraction",
            "status":       "success",
            "fields_found": fields_found,
            "field_count":  len(fields_found),
            "latency":      elapsed,
        }],
    }
