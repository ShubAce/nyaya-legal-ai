# Nyaya-7B: Indian Legal Judgment Intelligence System

> **Finetuned Mistral-7B model trained on Indian Supreme Court and High Court judgments that extracts parties, statutes, holdings, and precedents into structured JSON — wrapped in a LangGraph agentic pipeline.**

---

## Project Architecture Overview

```
Raw Judgment PDF / Text
        │
        ▼
┌───────────────────┐
│  Extraction Agent │  ◄── Nyaya-7B (your finetuned model)
│  (Nyaya-7B)       │
└────────┬──────────┘
         │ structured JSON (draft)
         ▼
┌───────────────────┐
│  Validation Agent │  ◄── RAG over IPC + CrPC + Constitution
│  (Statute RAG)    │       (ChromaDB)
└────────┬──────────┘
         │ validated + enriched JSON
         ▼
┌───────────────────┐
│  Precedent Agent  │  ◄── Citation resolver (IndianKanoon API
│  (Citation RAG)   │       + local Chroma index)
└────────┬──────────┘
         │ final enriched JSON
         ▼
┌───────────────────┐
│  Confidence Scorer│  ◄── Per-field uncertainty scoring
│  + Audit Trail    │       (Uncertainty analysis)
└────────┬──────────┘
         │
         ▼
   FastAPI Backend → Next.js Dashboard + HuggingFace Space
```

---

## Repository Structure

```
nyaya-legal-ai/
├── data/
│   ├── raw/                    # downloaded judgment text files
│   ├── processed/              # instruction-tuning JSONL
│   ├── statute_corpus/         # IPC, CrPC, Constitution text
│   └── test_set/               # 500 held-out judgments — NEVER TOUCH DURING TRAINING
├── finetuning/
│   ├── prepare_dataset.py      # data cleaning + formatting
│   ├── train.py                # QLoRA training script
│   ├── evaluate_model.py       # perplexity + JSON validity checks
│   └── merge_and_push.py       # merge LoRA weights + push to HF Hub
├── agents/
│   ├── graph.py                # LangGraph pipeline definition
│   ├── state.py                # shared LangGraph state schema
│   ├── extraction_agent.py     # calls Nyaya-7B
│   ├── rag_agent.py            # statute validation via ChromaDB
│   ├── citation_agent.py       # precedent resolver
│   └── confidence_agent.py     # per-field confidence scoring (NEW)
├── evaluation/
│   ├── benchmark.py            # runs all 4 systems on test set
│   ├── metrics.py              # F1, accuracy, hallucination rate
│   └── results/                # saved benchmark JSONs + charts
├── api/                        # FastAPI backend
│   ├── main.py
│   ├── tasks.py                # Celery task definitions
│   └── schemas.py
├── demo/                       # HuggingFace Gradio space
├── notebooks/                  # Kaggle training notebooks
├── docker-compose.yml
└── README.md
```

## Quick Start

1. **Install Base Dependencies:**
   ```bash
   pip install transformers==4.44.0 trl peft bitsandbytes accelerate \
               datasets sentencepiece wandb langchain langgraph \
               chromadb fastapi celery redis httpx gradio pydantic
   ```

2. **Run API Service:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

3. **Launch Gradio UI Demo:**
   ```bash
   python demo/app.py
   ```

4. **Docker Compose:**
   ```bash
   docker-compose up --build
   ```

---

## Benchmark Results (Expected)

| Metric | Mistral Base | **Nyaya-7B** | GPT-3.5 | GPT-4o |
|---|---|---|---|---|
| Statute F1 | 0.41 | **0.79** | 0.58 | 0.74 |
| Outcome accuracy | 0.52 | **0.88** | 0.71 | 0.85 |
| Party extraction | 0.61 | **0.91** | 0.74 | 0.89 |
| JSON validity | 67% | **98%** | 84% | 96% |
| Hallucination rate | 31% | **4%** | 18% | 8% |
| Cost per judgment | $0.00 | **$0.00** | $0.003 | $0.04 |
