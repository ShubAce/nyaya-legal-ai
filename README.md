# Nyaya-7B: Indian Legal Judgment Intelligence System

> **Finetuned Mistral-7B model trained on Indian Supreme Court and High Court judgments that extracts parties, statutes, holdings, and precedents into structured JSON — wrapped in a LangGraph agentic pipeline.**

---

## Project Architecture Overview

Everything runs locally: the finetuned model is loaded from disk onto your GPU
and nothing is downloaded from the Hugging Face Hub at runtime.

```
Raw Judgment PDF / Text
        │
        ▼
┌───────────────────┐
│  Extraction Agent │  ◄── Nyaya-7B (local merged checkpoint, 4-bit on GPU)
└────────┬──────────┘
         │ structured JSON (or → Assembly with an error if unparseable)
         ├──────────────────────────────┐
         ▼                              ▼
┌───────────────────┐        ┌───────────────────┐
│ Statute Validation│        │ Precedent Resolver│   (run in parallel)
│ grounded in the   │        │ local Chroma index│
│ judgment + corpus │        │ + text scan       │
└────────┬──────────┘        └────────┬──────────┘
         └──────────────┬─────────────┘
                        ▼
              ┌───────────────────┐
              │ Confidence Scorer │  ◄── Per-field uncertainty scoring
              └────────┬──────────┘
                       ▼
              ┌───────────────────┐
              │     Assembly      │  ──► FastAPI (SSE) ──► Streamlit UI
              └───────────────────┘
```

---

## Repository Structure

```
nyaya-legal-ai/
└── backend/
    ├── agents/                 # LangGraph pipeline
    │   ├── config.py           # paths, offline mode, cached ChromaDB collections
    │   ├── graph.py            # pipeline definition (+ CLI: python -m agents.graph)
    │   ├── state.py            # shared state schema
    │   ├── extraction_agent.py # runs Nyaya-7B locally
    │   ├── rag_agent.py        # statute validation
    │   ├── citation_agent.py   # precedent resolver
    │   └── confidence_agent.py # per-field confidence scoring
    ├── api/                    # FastAPI backend (main.py, schemas.py)
    ├── streamlit_app.py        # Streamlit UI (talks to the API)
    ├── data/                   # statute corpus, ChromaDB indexes, test set
    ├── kaggle/                 # local model checkpoints (git-ignored)
    ├── finetuning/             # dataset prep, QLoRA training, merge
    ├── notebooks/              # Kaggle training notebook
    ├── evaluation/             # benchmark, metrics, adversarial tests
    └── scripts/                # knowledge-base builders, active learning
```

## Quick Start (local)

Requirements: an NVIDIA GPU with at least 6 GB of VRAM, and the merged model at
`backend/kaggle/nyaya7b-finetuning-output/nyaya-checkpoints/nyaya-7b-merged`
(or set `FINETUNED_MODEL_PATH` in `backend/.env`).

1. **Install dependencies** (inside `backend/`):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

2. **Start the API and UI** — double-click `backend\run.bat`, or run it from `backend/`.
   It uses the project venv (`backend\.nyaya`) so the system/Anaconda Python is never used.
   The model loads onto the GPU in the background (~30–40 s). API docs: http://127.0.0.1:8000/docs

3. **Or start them by hand** — activate the venv first, otherwise `python` is Anaconda's:
   ```bat
   cd backend
   .nyaya\Scripts\activate
   python -m api.main                 &:: terminal 1
   streamlit run streamlit_app.py     &:: terminal 2 (activate the venv there too)
   ```

4. **Or run the pipeline straight from the command line:**
   ```bash
   python -m agents.graph                 # built-in sample judgment
   python -m agents.graph judgment.txt    # your own file
   ```

The embedding model (`BAAI/bge-base-en-v1.5`) must already be in your local
Hugging Face cache; it is read offline. Set `HF_HUB_OFFLINE=0` only if you need to
download it once. IndianKanoon lookups are off by default; set
`USE_INDIANKANOON=1` and `INDIANKANOON_API_KEY` in `backend/.env` to enable them.

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
