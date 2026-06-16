---
language:
- en
- hi
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
- legal
- indian-legal
- information-extraction
- nlp
- qlora
- peft
- finetuned
- mistral
- json-extraction
- legal-nlp
datasets:
- d0r1h/ILSum
- law-ai/InLegalNLP
pipeline_tag: text-generation
library_name: transformers
model-index:
- name: nyaya-7b
  results:
  - task:
      type: text-generation
      name: Structured Legal Information Extraction
    metrics:
    - type: f1
      value: 0.425
      name: Statute F1
    - type: accuracy
      value: 0.64
      name: Outcome Accuracy
    - type: other
      value: 0.427
      name: Hallucination Rate (lower is better)
    - type: other
      value: 0.86
      name: JSON Validity Rate
---

# 🏛️ Nyaya-7B — Indian Legal Judgment Parser

<p align="center">
  <img src="https://img.shields.io/badge/Base%20Model-Mistral--7B--Instruct--v0.3-blue" />
  <img src="https://img.shields.io/badge/Method-QLoRA%20(4--bit%20NF4)-orange" />
  <img src="https://img.shields.io/badge/Domain-Indian%20Legal%20NLP-green" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-red" />
  <img src="https://img.shields.io/badge/Language-English%20%7C%20Hindi-purple" />
</p>

**Nyaya-7B** is a domain-adapted, instruction-finetuned version of [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), trained on **10,000+ Indian Supreme Court and High Court judgments** to extract structured legal information into clean, validated JSON — at **zero API cost**, fully offline.

> **"Nyaya" (न्याय)** means *justice* in Sanskrit and Hindi.

---

## 🎯 What it does

Given raw Indian court judgment text, Nyaya-7B extracts a full structured JSON covering:

| Field | Description |
|---|---|
| `case_name` | Petitioner v. Respondent |
| `citation` | AIR / SCC / SCR citation |
| `court` | Full court name (Supreme Court, High Court, etc.) |
| `year` | Year of judgment |
| `petitioner` / `respondent` | Party names |
| `subject_matter` | Criminal / Civil / Constitutional / Tax / ... |
| `statutes_cited` | List of Acts + Sections + descriptions |
| `precedents_cited` | AIR/SCC citations with case names |
| `legal_issues` | Issues framed by the court |
| `holding` | Court's decision and reasoning (1–3 sentences) |
| `outcome` | dismissed / allowed / disposed / remanded / modified |

---

## 📊 Benchmark Results

Evaluated on a **50-case held-out test set** of Indian SC/HC judgments, benchmarked head-to-head against Gemini 2.5 Flash:

| Metric | Gemini 2.5 Flash | **Nyaya-7B** | Winner |
|---|---|---|---|
| **Statute F1** | 0.227 | **0.425** | 🏆 Nyaya-7B (+87%) |
| **Outcome Accuracy** | 0.20 | **0.64** | 🏆 Nyaya-7B (+220%) |
| **Hallucination Rate** ↓ | 0.775 | **0.427** | 🏆 Nyaya-7B (−45%) |
| **JSON Validity** | 0.98 | 0.86 | Gemini Flash |
| **Field Coverage** | 0.893 | 0.855 | Gemini Flash |
| **Cost per Judgment** | ~$0.0001 | **$0.00** | 🏆 Nyaya-7B |

> Nyaya-7B achieves **3.2× higher outcome classification accuracy** and **45% lower hallucination rate** than Gemini 2.5 Flash, while running entirely offline at zero cost.

---

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch accelerate bitsandbytes
```

### Load & Run (GPU, 4-bit quantized)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch, json

MODEL_ID = "your-username/nyaya-7b"   # replace with your HuggingFace repo

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

### Extract structured data from a judgment

```python
judgment_text = """
IN THE SUPREME COURT OF INDIA
Criminal Appeal No. 1234 of 2022

State of Punjab                         ...Appellant
Versus
Gurpreet Singh                          ...Respondent

JUDGMENT

The appellant challenges the High Court's order acquitting the respondent 
of charges under Section 302 IPC read with Section 34 IPC...
"""

messages = [
    {"role": "system", "content": "You are Nyaya, a specialized Indian legal extraction model."},
    {"role": "user", "content": f"Extract structured data from this judgment and return JSON:\n\n{judgment_text}"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

raw_output = pipe(
    prompt,
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
)[0]["generated_text"]

result = json.loads(raw_output.strip())
print(result)
```

### Expected Output

```json
{
  "case_name": "State of Punjab v. Gurpreet Singh",
  "citation": null,
  "court": "Supreme Court of India",
  "year": 2022,
  "petitioner": "State of Punjab",
  "respondent": "Gurpreet Singh",
  "subject_matter": "Criminal",
  "statutes_cited": [
    {"act": "Indian Penal Code", "section": "302", "description": "Punishment for murder"},
    {"act": "Indian Penal Code", "section": "34",  "description": "Acts done by several persons in furtherance of common intention"}
  ],
  "precedents_cited": [],
  "legal_issues": [
    "Whether the High Court was justified in acquitting the respondent under Section 302 IPC?"
  ],
  "holding": "The Supreme Court examined the evidence and found the High Court's reasoning sound...",
  "outcome": "dismissed"
}
```

### CPU Inference (no GPU required)

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
# Note: CPU inference is significantly slower (~5–15 min per judgment)
```

---

## 🔧 Training Details

| Parameter | Value |
|---|---|
| **Base model** | mistralai/Mistral-7B-Instruct-v0.3 |
| **Fine-tuning method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Quantization** | 4-bit NF4 with double quantization |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.05 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Training samples** | ~10,000 Indian SC/HC judgment pairs |
| **Epochs** | 3 |
| **Effective batch size** | 16 (batch 2 × grad_accum 8) |
| **Learning rate** | 2e-4 |
| **LR scheduler** | Cosine |
| **Optimizer** | paged_adamw_8bit |
| **Max sequence length** | 2048 tokens |
| **Hardware** | Kaggle T4 × 2 (32 GB VRAM total) |
| **Training time** | ~4 hours |
| **Compute dtype** | float16 |

### Training Infrastructure

- Fine-tuned on **Kaggle T4 × 2** GPU notebooks
- Monitored with **Weights & Biases** (JSON validity rate logged per epoch)
- Adapter merged into base model weights with `merge_and_unload()` for zero inference overhead

---

## 📚 Training Data

The model was trained on ~10,000 Indian court judgment pairs curated and labeled from:

- **[ILSum](https://huggingface.co/datasets/d0r1h/ILSum)** — Indian Legal Summarization dataset (Supreme Court judgments)
- **[InLegalNLP](https://huggingface.co/datasets/law-ai/InLegalNLP)** — Indian Legal NLP benchmark corpus

Labels were auto-generated using **Gemini 2.5 Flash** as a labelling oracle on the raw judgment texts, following the canonical extraction schema, then validated for JSON structure and field completeness.

---

## ⚠️ Limitations

- **Not for legal advice:** This model extracts structured information only. It does not provide legal opinions or advice. Always consult a qualified lawyer for legal matters.
- **Pre-1950 judgments:** May perform poorly on archaic legal language from older judgments.
- **Hindi/regional language text:** Primarily trained on English-language judgments; performance degrades on mixed-language or vernacular text.
- **Scanned/handwritten PDFs:** Model accepts only clean text input — OCR preprocessing is required for scanned documents.
- **Citation hallucination:** Significantly reduced (42.7% vs 77.5% baseline), but the model can still occasionally generate plausible-but-incorrect section numbers. Always validate critical citations against primary sources.
- **Novel statutes:** Statutes not well-represented in training data (e.g., recent 2023–24 acts) may have lower extraction accuracy.

---

## ✅ Intended Use

- ⚖️ Legal research and document processing automation
- 🤖 Paralegal workflow tools and legal analytics dashboards
- 📖 Academic research on Indian legal NLP
- 🔍 Building legal search and knowledge graph systems
- 📊 Bulk digitization of case records

## ❌ Out-of-Scope Use

- Providing legal advice to individuals
- Making or influencing judicial decisions
- Use in actual legal proceedings without qualified human review
- Any high-stakes decision-making without validation

---

## 📄 Output Schema

```python
{
  "case_name":         str,                          # "Petitioner v. Respondent"
  "citation":          str | None,                   # "AIR 1997 SC 3986" or null
  "court":             str,                          # Full court name
  "year":              int | None,                   # 4-digit year
  "petitioner":        str,
  "respondent":        str,
  "subject_matter":    str | None,                   # Criminal | Civil | Constitutional | ...
  "statutes_cited":    [{"act": str, "section": str, "description": str}],
  "precedents_cited":  [{"citation": str, "case_name": str | None}],
  "legal_issues":      [str],
  "holding":           str,                          # 1-3 sentence summary
  "outcome":           str                           # dismissed | allowed | disposed | remanded | modified
}
```

---

## 🔗 Related Resources

- 🐙 **GitHub:** [nyaya-legal-ai](https://github.com/your-username/nyaya-legal-ai) — Full project source code including FastAPI backend, LangGraph pipeline, and Streamlit UI
- 📊 **Base Model:** [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- 📁 **Dataset (ILSum):** [d0r1h/ILSum](https://huggingface.co/datasets/d0r1h/ILSum)

---

## 📝 Citation

If you use Nyaya-7B in your research or applications, please cite:

```bibtex
@misc{nyaya7b2026,
  title   = {Nyaya-7B: A QLoRA Fine-tuned LLM for Indian Legal Judgment Parsing},
  author  = {Your Name},
  year    = {2026},
  url     = {https://huggingface.co/your-username/nyaya-7b},
  note    = {Fine-tuned from mistralai/Mistral-7B-Instruct-v0.3 on 10,000+ Indian SC/HC judgments}
}
```

---

*Built with ❤️ for the Indian legal research community. Nyaya-7B is open-source and free to use under the Apache 2.0 license.*
