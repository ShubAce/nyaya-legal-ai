"""
api/main.py — FastAPI backend for the Nyaya Legal AI system.

Endpoints:
  POST /analyze           — upload PDF or text, stream agent events via SSE
  POST /analyze/text      — analyze raw text (JSON body)
  POST /compare           — run Nyaya-7B and Gemini side-by-side (needs Gemini keys)
  POST /similar           — find similar cases using vector similarity
  GET  /health            — health check with model status and index sizes
  GET  /audit/{job_id}    — retrieve full audit trail for a past job

Usage (from the backend/ directory):
    python -m api.main
    # or
    uvicorn api.main:app --port 8000
"""

import io
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import config  # noqa: F401  — offline mode + paths, before any model library loads
from agents.config import get_collection, loaded_collection
from agents.graph import build_pipeline, run_pipeline, stream_pipeline
from agents import extraction_agent

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from api.schemas import AnalyzeRequest, CompareResult, HealthResponse, SimilarCase

# ── Global state ──────────────────────────────────────────────────────────────
_pipeline = build_pipeline()
_job_store: dict[str, dict] = {}   # in-memory job store


def _warm_up_model() -> None:
    try:
        extraction_agent.load_model()
    except Exception as e:
        logger.error(f"Model warm-up failed: {e}")


def _warm_up_indexes() -> None:
    get_collection("statutes")
    get_collection("precedents")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model (GPU) and the indexes (CPU) in the background so the first request is fast.
    logger.info("Nyaya API starting — loading Nyaya-7B and indexes in the background...")
    threading.Thread(target=_warm_up_model, daemon=True).start()
    threading.Thread(target=_warm_up_indexes, daemon=True).start()
    yield


# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nyaya Legal AI API",
    description="Finetuned LLM system for Indian legal judgment parsing",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],   # Streamlit UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Text extraction from file ─────────────────────────────────────────────────

def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract text from uploaded PDF or plain text file."""
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise HTTPException(status_code=500, detail="pdfplumber not installed")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

    if filename_lower.endswith((".txt", ".text")):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .pdf or .txt file.")


# ── SSE streaming helper ──────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_analysis(judgment_text: str, job_id: str) -> Iterator[str]:
    """
    Yield SSE events as each agent completes. This is a plain (sync) generator,
    so Starlette runs it in a worker thread and model inference never blocks
    the event loop (health checks keep answering while a judgment is analysed).
    """
    job = _job_store[job_id] = {"status": "running", "started_at": time.time(), "events": []}
    errors: list[str] = []
    final_output = None

    try:
        yield _sse({"event": "start", "job_id": job_id})
        for event in stream_pipeline(_pipeline, judgment_text):
            job["events"].append(event)
            errors.extend(event.get("errors") or [])
            if event.get("final_output"):
                final_output = event["final_output"]
            yield _sse(event)
    except Exception as e:
        logger.exception(f"Pipeline error for job {job_id}")
        errors.append(str(e))

    job.update(final=final_output, errors=errors, finished_at=time.time())
    if final_output is None:
        job["status"] = "failed"
        message = errors[-1] if errors else "Pipeline produced no output"
        yield _sse({"event": "error", "job_id": job_id, "error": message, "errors": errors})
    else:
        job["status"] = "complete"
        yield _sse({"event": "complete", "job_id": job_id})


def _streaming_response(judgment_text: str) -> StreamingResponse:
    job_id = str(uuid.uuid4())
    logger.info(f"New job {job_id}: {len(judgment_text)} chars")
    return StreamingResponse(
        _stream_analysis(judgment_text, job_id),
        media_type="text/event-stream",
        headers={"X-Job-ID": job_id, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — reports whether the model and indexes are ready (never blocks on loading)."""
    statute_col   = loaded_collection("statutes")
    precedent_col = loaded_collection("precedents")
    return HealthResponse(
        status="ok",
        model_loaded=extraction_agent.is_model_loaded(),
        model_error=extraction_agent.model_load_error(),
        statute_index_size=statute_col.count() if statute_col else 0,
        precedent_index_size=precedent_col.count() if precedent_col else 0,
    )


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Upload a PDF or .txt judgment. Returns a Server-Sent Events stream of agent progress + final result."""
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    judgment_text = extract_text_from_upload(file_bytes, file.filename or "upload.txt")
    if len(judgment_text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Extracted text too short to be a valid judgment")
    return _streaming_response(judgment_text)


@app.post("/analyze/text")
def analyze_text(request: AnalyzeRequest):
    """Analyze raw judgment text. Streams if request.stream=True, else waits for the full result."""
    if len(request.judgment_text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Judgment text too short")

    if request.stream:
        return _streaming_response(request.judgment_text)

    result = run_pipeline(_pipeline, request.judgment_text)
    return JSONResponse(content={
        "job_id":       str(uuid.uuid4()),
        "final_output": result.get("final_output"),
        "audit_trail":  result.get("audit_trail"),
        "errors":       result.get("errors", []),
    })


@app.post("/compare")
def compare_systems(request: AnalyzeRequest):
    """
    Run Nyaya-7B and Gemini on the same text and return side-by-side results.
    Gemini is an online API and needs GEMINI_API_KEY_* in backend/.env.
    """
    from evaluation.benchmark import NyayaFinetunedSystem, GeminiProSystem

    text    = request.judgment_text[:6000]
    results = {}
    for name, system_cls in (("nyaya_7b", NyayaFinetunedSystem), ("gemini_pro", GeminiProSystem)):
        try:
            _, output = system_cls().predict(text)
            results[name] = output
        except Exception as e:
            results[name] = {"error": str(e)}

    return CompareResult(judgment_text_preview=text[:500], systems=results)


@app.post("/similar", response_model=list[SimilarCase])
def find_similar_cases(request: AnalyzeRequest, top_k: int = 5):
    """Find the top-k most similar cases from the precedent corpus."""
    collection = get_collection("precedents")
    if collection is None or collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Precedent index not available. Run scripts/build_knowledge_base.py first.",
        )

    results = collection.query(
        query_texts=[request.judgment_text[:2000]],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        SimilarCase(
            citation=meta.get("citation", ""),
            case_name=meta.get("case_name", ""),
            court=meta.get("court", ""),
            year=meta.get("year", 0),
            similarity_score=round(1 - dist, 4),   # cosine distance → similarity
            holding_preview=doc[:200],
        )
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


@app.get("/audit/{job_id}")
def get_audit(job_id: str):
    """Retrieve the full audit trail for a completed job."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_store[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", 8000)),
        workers=1,   # single worker — one copy of the model on the GPU
    )
