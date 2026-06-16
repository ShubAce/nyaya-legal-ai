"""
build_knowledge_base.py — Phase 3: Build ChromaDB indexes for statute validation
and precedent resolution.

Two collections are built:
  1. statutes   — IPC, CrPC, Constitution, Evidence Act, etc. (section-level chunks)
  2. precedents — one entry per judgment from training corpus (for similarity search)

Usage:
    python scripts/build_knowledge_base.py
"""

import os
import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

print("starting")

CHROMA_STATUTE_PATH   = Path(os.getenv("CHROMA_STATUTE_PATH",   "data/chroma_statutes"))
CHROMA_PRECEDENT_PATH = Path(os.getenv("CHROMA_PRECEDENT_PATH", "data/chroma_precedents"))
STATUTE_CORPUS_DIR    = Path("data/statute_corpus")
TRAIN_DATA_PATH       = Path("data/processed/train.jsonl")

# Embedding model — strong, fast, free
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# ── Statute texts ─────────────────────────────────────────────────────────────
# These are downloaded from India Code (https://www.indiacode.nic.in)
# and saved as plain text files in data/statute_corpus/
STATUTE_FILES = {
    "Indian Penal Code":              "ipc.txt",
    "Code of Criminal Procedure":     "crpc.txt",
    "Constitution of India":          "constitution.txt",
    "Indian Evidence Act":            "evidence_act.txt",
    "Code of Civil Procedure":        "cpc.txt",
    "Negotiable Instruments Act":     "negotiable_instruments_act.txt",
    "Transfer of Property Act":       "transfer_of_property_act.txt",
    "Contract Act":                   "contract_act.txt",
    "Limitation Act":                 "limitation_act.txt",
    "Prevention of Corruption Act":   "prevention_of_corruption_act.txt",
}

# ── Statute section patterns ───────────────────────────────────────────────────
# Used to split statute text into section-level chunks
SECTION_PATTERNS = [
    r"(?=\n\s*Section\s+\d+[\.\-])",        # "Section 302."
    r"(?=\n\s*\d+[\.\)]\s+[A-Z])",          # "302. Murder."
    r"(?=\nARTICLE\s+\d+)",                 # "ARTICLE 21"
    r"(?=\n\s*Order\s+[IVXLCDM]+\s*[\.\-])", # CPC Orders
]


def get_embedding_function():
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


# ── Build statute index ───────────────────────────────────────────────────────

def parse_statute_sections(text: str, act_name: str) -> list[dict]:
    """
    Split a statute text into individual sections/articles.
    Returns list of {id, text, section_number, act} dicts.
    """
    # Try section-level splitting first
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\nSection", "\n\nARTICLE", "\n\n", "\n", ". "],
    )
    chunks = splitter.split_text(text)

    sections = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 50:
            continue

        # Try to extract section number from chunk
        sec_match = re.search(r"(?:Section|ARTICLE|Order)\s+(\d+[A-Za-z]*)", chunk, re.IGNORECASE)
        sec_num = sec_match.group(1) if sec_match else str(i)

        sections.append({
            "id":             f"{act_name.replace(' ', '_')}_{sec_num}_{i}",
            "text":           chunk,
            "section_number": sec_num,
            "act":            act_name,
        })

    return sections


def build_statute_index(client: chromadb.ClientAPI, ef) -> chromadb.Collection:
    collection = client.get_or_create_collection(
        name="statutes",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Skip if already populated
    if collection.count() > 100:
        logger.info(f"Statute index already has {collection.count()} chunks — skipping rebuild")
        return collection

    logger.info("Building statute index...")
    total_chunks = 0

    for act_name, filename in STATUTE_FILES.items():
        filepath = STATUTE_CORPUS_DIR / filename
        if not filepath.exists():
            logger.warning(f"Statute file not found: {filepath} — download from indiacode.nic.in")
            # Create a minimal placeholder so the system still works
            _create_placeholder_statute(filepath, act_name)

        text = filepath.read_text(encoding="utf-8", errors="ignore")
        sections = parse_statute_sections(text, act_name)

        if not sections:
            logger.warning(f"No sections parsed from {filename}")
            continue

        # Batch upsert to ChromaDB
        batch_size = 100
        for i in range(0, len(sections), batch_size):
            batch = sections[i:i + batch_size]
            collection.upsert(
                ids=[s["id"] for s in batch],
                documents=[s["text"] for s in batch],
                metadatas=[{"act": s["act"], "section": s["section_number"]} for s in batch],
            )

        total_chunks += len(sections)
        logger.success(f"  {act_name}: {len(sections)} chunks indexed")

    logger.success(f"Statute index complete: {total_chunks} total chunks")
    return collection


def _create_placeholder_statute(filepath: Path, act_name: str):
    """
    Creates a minimal placeholder statute file with common sections.
    Replace with the actual downloaded text from indiacode.nic.in
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    placeholder = f"""# {act_name}
# PLACEHOLDER — Replace this file with the actual text from https://www.indiacode.nic.in

Section 1. Short title, extent and commencement.
This Act may be called the {act_name}.

Section 2. Definitions.
In this Act, unless the context otherwise requires...

# Download the full text from:
# https://www.indiacode.nic.in/handle/123456789/1362
"""
    filepath.write_text(placeholder)
    logger.warning(f"Created placeholder for {act_name} at {filepath}")
    logger.warning(f"Download the full text from https://www.indiacode.nic.in for accurate validation")


# ── Build precedent index ─────────────────────────────────────────────────────

def build_precedent_index(client: chromadb.ClientAPI, ef) -> chromadb.Collection:
    collection = client.get_or_create_collection(
        name="precedents",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    if not TRAIN_DATA_PATH.exists():
        logger.warning(f"Training data not found at {TRAIN_DATA_PATH} — precedent index will be empty")
        return collection

    if collection.count() > 100:
        logger.info(f"Precedent index already has {collection.count()} entries — skipping rebuild")
        return collection

    logger.info("Building precedent index from training corpus...")

    ids, documents, metadatas = [], [], []

    with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Indexing precedents"):
            try:
                row = json.loads(line)
                messages = row.get("messages", [])
                label_msg = next((m for m in messages if m["role"] == "assistant"), None)
                if not label_msg:
                    continue

                label = json.loads(label_msg["content"])
                citation  = label.get("citation", "")
                case_name = label.get("case_name", "")
                holding   = label.get("holding", "")
                year      = label.get("year", 0)
                court     = label.get("court", "")
                outcome   = label.get("outcome", "")

                if not citation or not case_name or not holding:
                    continue

                # Document = case name + holding — this is what gets embedded
                document = f"{case_name}\n{holding}"

                ids.append(citation.replace(" ", "_")[:100])
                documents.append(document)
                metadatas.append({
                    "citation":  citation,
                    "case_name": case_name,
                    "year":      year or 0,
                    "court":     court,
                    "outcome":   outcome,
                })

            except (json.JSONDecodeError, KeyError):
                continue

    # Batch upsert
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    logger.success(f"Precedent index complete: {len(ids)} cases indexed")
    return collection


# ── Verify indexes ────────────────────────────────────────────────────────────

def verify_indexes(statute_col: chromadb.Collection, precedent_col: chromadb.Collection):
    """Run a few test queries to verify both indexes work correctly."""
    logger.info("\nVerifying indexes with test queries...")

    # Test 1: Statute lookup — IPC 302
    results = statute_col.query(
        query_texts=["Indian Penal Code Section 302 murder"],
        n_results=3,
    )
    logger.info("\nTest query: 'IPC Section 302 murder'")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        logger.info(f"  [{dist:.3f}] {meta['act']} §{meta['section']}: {doc[:80]}...")

    # Test 2: Precedent lookup
    if precedent_col.count() > 0:
        results = precedent_col.query(
            query_texts=["circumstantial evidence murder conviction"],
            n_results=3,
        )
        logger.info("\nTest query: 'circumstantial evidence murder conviction'")
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            logger.info(f"  [{dist:.3f}] {meta.get('citation', 'N/A')} — {meta.get('case_name', '')[:60]}")

    logger.success("Index verification complete")

print("running main")

def main():
    logger.info("=" * 60)
    logger.info("Nyaya-7B — Phase 3: Building Knowledge Bases")
    logger.info("=" * 60)

    # Ensure directories exist
    CHROMA_STATUTE_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_PRECEDENT_PATH.mkdir(parents=True, exist_ok=True)
    STATUTE_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize embedding function (shared across both indexes)
    ef = get_embedding_function()

    # Statute index
    statute_client = chromadb.PersistentClient(path=str(CHROMA_STATUTE_PATH))
    statute_col    = build_statute_index(statute_client, ef)

    # Precedent index
    precedent_client = chromadb.PersistentClient(path=str(CHROMA_PRECEDENT_PATH))
    precedent_col    = build_precedent_index(precedent_client, ef)

    # Verify
    verify_indexes(statute_col, precedent_col)

    logger.info("")
    logger.success(f"Statute index:   {statute_col.count()} chunks  @ {CHROMA_STATUTE_PATH}")
    logger.success(f"Precedent index: {precedent_col.count()} cases @ {CHROMA_PRECEDENT_PATH}")
    logger.info("Phase 3 complete. Next: run agents/graph.py")


if __name__ == "__main__":
    main()
