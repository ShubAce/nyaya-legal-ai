"""
api/main.py — Phase 7: FastAPI backend for the Nyaya Legal AI system.

Endpoints:
  POST /analyze           — upload PDF or text, stream agent events via SSE
  POST /analyze/text      — analyze raw text (JSON body)
  POST /compare           — run all 4 systems side-by-side
  POST /similar           — find similar cases using vector similarity
  GET  /health            — health check with index sizes
  GET  /audit/{job_id}    — retrieve full audit trail for a past job

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import uuid
import time
from pathlib import Path
from typing import AsyncIterator

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import (
    AnalyzeRequest, AnalysisResult, CompareResult,
    SimilarCase, HealthResponse, StreamEvent,
)
from agents.graph import build_pipeline, run_pipeline, stream_pipeline

load_dotenv()

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nyaya Legal AI API",
    description="Finetuned LLM system for Indian legal judgment parsing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",   # Streamlit UI
        "http://127.0.0.1:8501",
        "https://your-deployed-frontend.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_pipeline  = None
_job_store: dict[str, dict] = {}   # in-memory job store (replace with Redis for prod)

# Resolve paths relative to the backend directory (not cwd)
_BACKEND_DIR = Path(__file__).resolve().parent.parent

CHROMA_STATUTE_PATH   = _BACKEND_DIR / Path(os.getenv("CHROMA_STATUTE_PATH",   "data/chroma_statutes"))
CHROMA_PRECEDENT_PATH = _BACKEND_DIR / Path(os.getenv("CHROMA_PRECEDENT_PATH", "data/chroma_precedents"))
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising LangGraph pipeline...")
        _pipeline = build_pipeline()
        logger.success("Pipeline ready")
    return _pipeline


def get_precedent_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_PRECEDENT_PATH))
    try:
        return client.get_collection("precedents", embedding_function=ef)
    except Exception:
        return None


def get_statute_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_STATUTE_PATH))
    try:
        return client.get_collection("statutes", embedding_function=ef)
    except Exception:
        return None


# ── Text extraction from file ─────────────────────────────────────────────────

def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract text from uploaded PDF or plain text file."""
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages)
        except ImportError:
            raise HTTPException(status_code=500, detail="pdfplumber not installed")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

    elif filename_lower.endswith((".txt", ".text")):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .pdf or .txt file."
        )


# ── SSE streaming helper ──────────────────────────────────────────────────────

async def _stream_analysis(judgment_text: str, job_id: str) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted events as each agent completes.
    """
    pipeline = get_pipeline()
    _job_store[job_id] = {"status": "running", "started_at": time.time(), "events": []}

    try:
        # Signal start
        yield f"data: {json.dumps({'event': 'start', 'job_id': job_id})}\n\n"

        for event in stream_pipeline(pipeline, judgment_text):
            _job_store[job_id]["events"].append(event)
            yield f"data: {json.dumps(event)}\n\n"

        # Final result is in the last event
        final_event = _job_store[job_id]["events"][-1] if _job_store[job_id]["events"] else {}
        _job_store[job_id]["status"]     = "complete"
        _job_store[job_id]["final"]      = final_event.get("final_output")
        _job_store[job_id]["finished_at"] = time.time()

        yield f"data: {json.dumps({'event': 'complete', 'job_id': job_id})}\n\n"

    except Exception as e:
        error_event = {"event": "error", "job_id": job_id, "error": str(e)}
        _job_store[job_id]["status"] = "failed"
        _job_store[job_id]["error"]  = str(e)
        logger.error(f"Pipeline error for job {job_id}: {e}")
        yield f"data: {json.dumps(error_event)}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verifies model and indexes are ready."""
    statute_col   = get_statute_collection()
    precedent_col = get_precedent_collection()
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        statute_index_size=statute_col.count() if statute_col else 0,
        precedent_index_size=precedent_col.count() if precedent_col else 0,
    )


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Upload a PDF or .txt judgment file.
    Returns a Server-Sent Events stream of agent progress + final result.
    """
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    judgment_text = extract_text_from_upload(file_bytes, file.filename or "upload.txt")
    if len(judgment_text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Extracted text too short to be a valid judgment")

    job_id = str(uuid.uuid4())
    logger.info(f"New job {job_id}: {len(judgment_text)} chars from {file.filename}")

    return StreamingResponse(
        _stream_analysis(judgment_text, job_id),
        media_type="text/event-stream",
        headers={
            "X-Job-ID":          job_id,
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
        },
    )


@app.post("/analyze/text")
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze raw judgment text (JSON body).
    Returns stream if request.stream=True, else waits for full result.
    """
    if len(request.judgment_text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Judgment text too short")

    job_id = str(uuid.uuid4())

    if request.stream:
        return StreamingResponse(
            _stream_analysis(request.judgment_text, job_id),
            media_type="text/event-stream",
            headers={"X-Job-ID": job_id, "Cache-Control": "no-cache"},
        )
    else:
        # Blocking mode — wait for full result
        pipeline = get_pipeline()
        result   = run_pipeline(pipeline, request.judgment_text)
        return JSONResponse(content={
            "job_id":       job_id,
            "final_output": result.get("final_output"),
            "audit_trail":  result.get("audit_trail"),
            "errors":       result.get("errors", []),
        })


@app.post("/compare")
async def compare_systems(request: AnalyzeRequest):
    """
    Run Nyaya-7B and Gemini 1.5 Pro on the same text and return side-by-side results.
    Used by the diff view in the frontend.
    """
    from evaluation.benchmark import NyayaFinetunedSystem, GeminiProSystem

    text    = request.judgment_text[:6000]
    results = {}

    # Nyaya-7B
    try:
        nyaya  = NyayaFinetunedSystem()
        _, nyaya_out = nyaya.predict(text)
        results["nyaya_7b"] = nyaya_out
    except Exception as e:
        results["nyaya_7b"] = {"error": str(e)}

    # Gemini 1.5 Pro
    try:
        gemini_pro = GeminiProSystem()
        _, gemini_out = gemini_pro.predict(text)
        results["gemini_pro"] = gemini_out
    except Exception as e:
        results["gemini_pro"] = {"error": str(e)}

    return CompareResult(
        judgment_text_preview=text[:500],
        systems=results,
    )


@app.post("/similar", response_model=list[SimilarCase])
async def find_similar_cases(request: AnalyzeRequest, top_k: int = 5):
    """
    Find the top-k most similar cases from the precedent corpus.
    Uses semantic similarity on the judgment text.
    """
    collection = get_precedent_collection()
    if collection is None or collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Precedent index not available. Run scripts/build_knowledge_base.py first."
        )

    query_text = request.judgment_text[:2000]
    results    = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    similar_cases = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similar_cases.append(SimilarCase(
            citation=       meta.get("citation", ""),
            case_name=      meta.get("case_name", ""),
            court=          meta.get("court", ""),
            year=           meta.get("year", 0),
            similarity_score=round(1 - dist, 4),   # cosine distance → similarity
            holding_preview=doc[:200],
        ))

    return similar_cases


@app.get("/audit/{job_id}")
async def get_audit(job_id: str):
    """Retrieve the full audit trail for a completed job."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_store[job_id]


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-load the pipeline on startup so the first request isn't slow."""
    logger.info("Nyaya API starting up...")
    try:
        get_pipeline()
        logger.success("Pipeline pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Pipeline pre-load failed (will retry on first request): {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
        workers=1,   # single worker — model is too large to fork
    )
