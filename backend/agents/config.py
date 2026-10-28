"""
config.py — Shared settings and cached resources for the Nyaya pipeline.

Everything runs locally: the finetuned model and the embedding model are read
from disk, and the Hugging Face Hub is forced into offline mode so nothing is
downloaded at runtime. Relative paths are resolved against the backend/
directory, so the pipeline works no matter which directory it is started from.
"""

import os
import threading
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# Must be set before transformers / sentence-transformers / datasets are imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _backend_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND_DIR / path


MODEL_PATH = _backend_path(os.getenv(
    "FINETUNED_MODEL_PATH",
    "kaggle/nyaya7b-finetuning-output/nyaya-checkpoints/nyaya-7b-merged",
))
CHROMA_STATUTE_PATH   = _backend_path(os.getenv("CHROMA_STATUTE_PATH",   "data/chroma_statutes"))
CHROMA_PRECEDENT_PATH = _backend_path(os.getenv("CHROMA_PRECEDENT_PATH", "data/chroma_precedents"))
EMBED_MODEL           = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))

# IndianKanoon is a paid online API, so it is only used when explicitly enabled.
INDIANKANOON_API_KEY = os.getenv("INDIANKANOON_API_KEY") or os.getenv("INDIAN_KANOON_API_KEY", "")
USE_INDIANKANOON     = os.getenv("USE_INDIANKANOON", "0") == "1" and bool(INDIANKANOON_API_KEY)


# ── Cached resources ──────────────────────────────────────────────────────────
# The statute and precedent agents run in parallel, so loading is guarded by a
# lock to stop both branches from loading the embedding model at the same time.

_resource_lock = threading.Lock()

_COLLECTION_PATHS = {
    "statutes":   CHROMA_STATUTE_PATH,
    "precedents": CHROMA_PRECEDENT_PATH,
}


@lru_cache(maxsize=1)
def _embedding_function():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


_collections: dict = {}


def _open_collection(name: str):
    import chromadb
    from loguru import logger

    path = _COLLECTION_PATHS[name]
    if not path.exists():
        logger.warning(f"ChromaDB '{name}' index not found at {path}. Run scripts/build_knowledge_base.py.")
        return None
    try:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name, embedding_function=_embedding_function())
    except Exception as e:
        logger.warning(f"Could not open ChromaDB '{name}' collection at {path}: {e}")
        return None
    logger.success(f"Loaded '{name}' collection: {collection.count()} entries")
    return collection


def get_collection(name: str):
    """Return the cached 'statutes' or 'precedents' collection, or None if unavailable."""
    if name not in _collections:
        with _resource_lock:
            if name not in _collections:
                _collections[name] = _open_collection(name)
    return _collections[name]


def loaded_collection(name: str):
    """Return the collection only if it is already loaded — never blocks or loads."""
    return _collections.get(name)
