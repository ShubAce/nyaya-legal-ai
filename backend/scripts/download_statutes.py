"""
download_statutes.py — Download Indian statute texts from HuggingFace and
build ChromaDB indexes for statute validation and precedent resolution.

Confirmed working datasets (verified June 2026):
  IPC sections:     harshitv804/Indian_Penal_Code
                    sairamn/indian-penal-code
                    karan842/ipc-sections
                    Exploration-Lab/IL-TUR (lsi config has statute text)
  Constitution:     nisaar/Constitution_of_India
                    Sharathhebbar24/Indian-Constitution
  Indian laws:      mratanusarkar/Indian-Laws  (bare acts — IPC, CrPC, Evidence Act etc.)
  Precedents:       opennyaiorg/InJudgements_dataset

Usage:
    python scripts/download_statutes.py
"""

import os
import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STATUTE_DIR           = Path("data/statute_corpus")
CHROMA_STATUTE_PATH   = Path(os.getenv("CHROMA_STATUTE_PATH",   "data/chroma_statutes"))
CHROMA_PRECEDENT_PATH = Path(os.getenv("CHROMA_PRECEDENT_PATH", "data/chroma_precedents"))
TRAIN_DATA_PATH       = Path("data/processed/train.jsonl")

for d in [STATUTE_DIR, CHROMA_STATUTE_PATH, CHROMA_PRECEDENT_PATH]:
    d.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-base-en-v1.5"


# ── Helper: check if HF dataset exists before loading ────────────────────────

def dataset_exists(dataset_id: str) -> bool:
    """Check if a HuggingFace dataset exists without downloading it."""
    try:
        from huggingface_hub import dataset_info
        dataset_info(dataset_id)
        return True
    except Exception:
        return False


def try_load_dataset(dataset_id: str, config=None, split="train", **kwargs):
    """Load a dataset, return None if it fails."""
    try:
        from datasets import load_dataset
        if config:
            ds = load_dataset(dataset_id, config, split=split,
                              trust_remote_code=False, **kwargs)
        else:
            ds = load_dataset(dataset_id, split=split,
                              trust_remote_code=False, **kwargs)
        logger.success(f"✓ Loaded {dataset_id} — {len(ds)} rows, columns: {ds.column_names}")
        return ds
    except Exception as e:
        logger.warning(f"✗ {dataset_id}: {e}")
        return None


# ── Step 1: Download statute texts ────────────────────────────────────────────

def clean_section_num(sec: str) -> str:
    s = sec.strip()
    # Remove common prefixes
    for prefix in ["IPC Section ", "IPC Section", "IPC_", "IPC ", "Section ", "Section", "Article ", "Article"]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s


def download_ipc() -> str:
    """Download IPC sections. Returns combined text."""
    logger.info("Downloading IPC...")
    sections = {}

    # Source 1: mratanusarkar/Indian-Laws — has bare acts including IPC
    ds = try_load_dataset("mratanusarkar/Indian-Laws")
    if ds:
        logger.info(f"  Indian-Laws columns: {ds.column_names}")
        for row in ds:
            # Log first row to see structure
            if not sections:
                for k, v in row.items():
                    if isinstance(v, str) and len(v) > 20:
                        logger.info(f"  [{k}]: {v[:80]}")
            act = str(row.get("act_title", "")).lower()
            if "penal" in act or "ipc" in act:
                sec_num = clean_section_num(str(row.get("section", "")))
                sec_text = str(row.get("law", ""))
                if sec_num and sec_text and len(sec_text) > 20:
                    sections[sec_num] = sec_text.strip()

    # Source 2: harshitv804/Indian_Penal_Code
    ds2 = try_load_dataset("harshitv804/Indian_Penal_Code")
    if ds2:
        for row in ds2:
            sec_num  = clean_section_num(str(row.get("section", row.get("Section",
                          row.get("section_number", "")))))
            sec_text = (row.get("section_text", "") or row.get("text", "") or
                       row.get("description", "") or row.get("content", "") or
                       row.get("Section_Desc", "") or row.get("Punishment", ""))
            if sec_num and sec_text and len(sec_text) > 20:
                sections[sec_num] = sec_text.strip()

    # Source 3: sairamn/indian-penal-code
    ds3 = try_load_dataset("sairamn/indian-penal-code")
    if ds3:
        for row in ds3:
            sec_num  = clean_section_num(str(row.get("section", row.get("Section", ""))))
            sec_text = (row.get("text", "") or row.get("description", "") or
                       row.get("content", "") or row.get("section_text", ""))
            if sec_num and sec_text and len(sec_text) > 20:
                sections[sec_num] = sec_text.strip()

    # Source 4: karan842/ipc-sections
    ds4 = try_load_dataset("karan842/ipc-sections")
    if ds4:
        for row in ds4:
            sec_num  = clean_section_num(str(row.get("section", row.get("Section", ""))))
            sec_text = (row.get("Description", "") or row.get("description", "") or
                        row.get("text", "") or row.get("Offense", "") or row.get("Punishment", ""))
            
            # Combine offense + punishment if both exist and description is empty
            if not (row.get("Description") or row.get("description") or row.get("text")):
                combined = ""
                if row.get("Offense"):
                    combined += f"Offense: {row['Offense']} "
                if row.get("Punishment"):
                    combined += f"Punishment: {row['Punishment']}"
                if combined:
                    sec_text = combined.strip()
                    
            if sec_num and sec_text and len(sec_text) > 10:
                sections[sec_num] = sec_text.strip()

    # Source 5: IL-TUR lsi config has IPC/other common statute text
    ds5 = try_load_dataset("Exploration-Lab/IL-TUR", config="lsi", split="statutes")
    if ds5:
        logger.info(f"  IL-TUR/lsi statutes columns: {ds5.column_names}")
        for row in ds5:
            sec_id   = str(row.get("id", ""))
            sec_num  = clean_section_num(sec_id)
            sec_text = row.get("text", "")
            if isinstance(sec_text, list):
                sec_text = " ".join(sec_text)
            if sec_num and sec_text and len(sec_text) > 20:
                sections[sec_num] = sec_text.strip()

    logger.info(f"IPC: {len(sections)} sections collected")

    # Build full text
    lines = ["INDIAN PENAL CODE, 1860\n"]
    
    # Sort keys numerically if possible
    def get_sort_key(x):
        import re
        match = re.match(r"^(\d+)", x)
        if match:
            return (int(match.group(1)), x)
        return (999999, x)

    for sec_num in sorted(sections.keys(), key=get_sort_key):
        lines.append(f"Section {sec_num}. {sections[sec_num]}")
    return "\n\n".join(lines)


def download_constitution() -> str:
    """Download Constitution of India articles."""
    logger.info("Downloading Constitution of India...")
    articles = {}

    # Source 1: Sharathhebbar24/Indian-Constitution
    ds = try_load_dataset("Sharathhebbar24/Indian-Constitution")
    if ds:
        logger.info(f"  Sharath Constitution columns: {ds.column_names}")
        for row in ds:
            art_id = str(row.get("article_id", ""))
            art_desc = row.get("article_desc", "")
            if art_id and art_desc:
                import re
                match = re.search(r"Article\s+(\d+[A-Z]*)", art_id, re.IGNORECASE)
                art_num = match.group(1) if match else art_id.strip()
                if art_num and art_desc and len(art_desc) > 20:
                    articles[art_num] = art_desc.strip()

    # Source 2: nisaar/Constitution_of_India
    if len(articles) < 50:
        ds2 = try_load_dataset("nisaar/Constitution_of_India")
        if ds2:
            logger.info(f"  Constitution columns: {ds2.column_names}")
            for row in ds2:
                art_num  = str(row.get("article", row.get("Article",
                              row.get("article_number", row.get("id", "")))))
                art_text = (row.get("text", "") or row.get("article_text", "") or
                           row.get("content", "") or row.get("description", ""))
                if not art_text and row.get("answer"):
                    art_text = row.get("answer", "")
                if not art_num and row.get("question"):
                    import re
                    match = re.search(r"Article\s+(\d+[A-Z]*)", row.get("question", ""), re.IGNORECASE)
                    art_num = match.group(1) if match else ""
                if art_num and art_text and len(art_text) > 20:
                    articles[art_num] = art_text.strip()

    # Source 3: moonmelonpizza/constitution_of_india
    if len(articles) < 50:
        ds3 = try_load_dataset("moonmelonpizza/constitution_of_india")
        if ds3:
            logger.info(f"  moonmelonpizza Constitution columns: {ds3.column_names}")
            current_art = None
            current_text = []
            for row in ds3:
                line = row.get("THE CONSTITUTION OF INDIA")
                if not line:
                    continue
                import re
                match = re.match(r"^Article\s+(\d+[A-Z]*)[\.\s-]", line, re.IGNORECASE)
                if match:
                    if current_art and current_text:
                        articles[current_art] = "\n".join(current_text)
                    current_art = match.group(1)
                    current_text = [line]
                elif current_art:
                    current_text.append(line)
            if current_art and current_text:
                articles[current_art] = "\n".join(current_text)

    logger.info(f"Constitution: {len(articles)} articles collected")

    lines = ["CONSTITUTION OF INDIA\n"]
    for art_num in sorted(articles.keys(), key=lambda x: float(x.replace("A","").replace("B","").replace("C","")[:6]) if x.replace("A","").replace("B","").replace("C","")[:6].replace(".","").isdigit() else 999):
        lines.append(f"Article {art_num}. {articles[art_num]}")
    return "\n\n".join(lines)


def download_other_acts() -> dict[str, str]:
    """Download CrPC, Evidence Act etc. from mratanusarkar/Indian-Laws."""
    logger.info("Downloading other acts (CrPC, Evidence Act, etc.)...")
    acts = {}

    ds = try_load_dataset("mratanusarkar/Indian-Laws")
    if not ds:
        logger.warning("Indian-Laws dataset not available — skipping CrPC/Evidence Act")
        return acts

    logger.info(f"  Indian-Laws columns: {ds.column_names}")
    if len(ds) > 0:
        for k, v in ds[0].items():
            if isinstance(v, str):
                logger.info(f"  [{k}]: {v[:80]}")

    # Group rows by act
    act_sections: dict[str, list] = {}
    for row in ds:
        act_name = str(row.get("act_title", "unknown")).strip()
        sec_num  = str(row.get("section", ""))
        sec_text = str(row.get("law", ""))
        if sec_text and len(sec_text) > 20:
            if act_name not in act_sections:
                act_sections[act_name] = []
            act_sections[act_name].append(f"Section {sec_num}. {sec_text}")

    # Map act names to output filenames
    act_file_map = {
        "crpc.txt":           ["criminal procedure", "crpc", "cr.p.c"],
        "evidence_act.txt":   ["evidence act", "evidence", "iea"],
        "cpc.txt":            ["civil procedure", "cpc", "c.p.c"],
        "contract_act.txt":   ["contract act"],
        "limitation_act.txt": ["limitation act"],
        "negotiable_instruments_act.txt": ["negotiable instruments"],
    }

    for filename, keywords in act_file_map.items():
        matching_acts = []
        for act_name, sections in act_sections.items():
            act_lower = act_name.lower()
            if any(kw in act_lower for kw in keywords):
                is_amendment = "amendment" in act_lower
                matching_acts.append((act_name, sections, is_amendment, len(sections)))
                
        if matching_acts:
            # Sort matching acts by:
            # 1. Non-amendment first
            # 2. Number of sections descending
            matching_acts.sort(key=lambda x: (not x[2], x[3]), reverse=True)
            best_act_name, best_sections, _, _ = matching_acts[0]
            text = f"{best_act_name.upper()}\n\n" + "\n\n".join(best_sections)
            acts[filename] = text
            logger.success(f"{filename}: {len(best_sections)} sections from '{best_act_name}'")

    logger.info(f"Other acts downloaded: {list(acts.keys())}")
    return acts


# ── Step 2: Save statute texts ────────────────────────────────────────────────

def save_statutes(ipc_text: str, constitution_text: str, other_acts: dict):
    """Save all statute texts to data/statute_corpus/."""
    saved = []

    if ipc_text and len(ipc_text) > 500:
        path = STATUTE_DIR / "ipc.txt"
        path.write_text(ipc_text, encoding="utf-8")
        logger.success(f"Saved ipc.txt ({len(ipc_text):,} chars)")
        saved.append("ipc.txt")

    if constitution_text and len(constitution_text) > 500:
        path = STATUTE_DIR / "constitution.txt"
        path.write_text(constitution_text, encoding="utf-8")
        logger.success(f"Saved constitution.txt ({len(constitution_text):,} chars)")
        saved.append("constitution.txt")

    for filename, text in other_acts.items():
        if text and len(text) > 200:
            path = STATUTE_DIR / filename
            path.write_text(text, encoding="utf-8")
            logger.success(f"Saved {filename} ({len(text):,} chars)")
            saved.append(filename)

    logger.info(f"Total statute files saved: {len(saved)} — {saved}")
    return saved


# ── Step 3: Build ChromaDB statute index ──────────────────────────────────────

def build_statute_index(ef) -> chromadb.Collection:
    """Build ChromaDB index from saved statute text files."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    client     = chromadb.PersistentClient(path=str(CHROMA_STATUTE_PATH))
    collection = client.get_or_create_collection(
        name="statutes",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 100:
        logger.info(f"Statute index already has {collection.count()} chunks — skipping rebuild")
        return collection

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\nSection", "\n\nArticle", "\n\nOrder", "\n\n", "\n"],
    )

    total_chunks = 0
    statute_files = list(STATUTE_DIR.glob("*.txt"))

    for filepath in statute_files:
        text = filepath.read_text(encoding="utf-8", errors="ignore")

        # Skip placeholder files
        if "PLACEHOLDER" in text or len(text) < 500:
            logger.warning(f"Skipping placeholder: {filepath.name}")
            continue

        act_name = filepath.stem.replace("_", " ").title()
        chunks   = splitter.split_text(text)
        if not chunks:
            continue

        # Batch upsert
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.upsert(
                ids=[f"{filepath.stem}_{i+j}" for j in range(len(batch))],
                documents=batch,
                metadatas=[{"act": act_name, "chunk_id": i+j,
                            "source": filepath.name} for j in range(len(batch))],
            )
        total_chunks += len(chunks)
        logger.success(f"  {filepath.name}: {len(chunks)} chunks indexed")

    logger.success(f"Statute index complete: {total_chunks} chunks total")
    return collection


# ── Step 4: Build ChromaDB precedent index ────────────────────────────────────

def build_precedent_index(ef) -> chromadb.Collection:
    """Build ChromaDB index from training corpus judgments."""
    client     = chromadb.PersistentClient(path=str(CHROMA_PRECEDENT_PATH))
    collection = client.get_or_create_collection(
        name="precedents",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 100:
        logger.info(f"Precedent index already has {collection.count()} entries — skipping")
        return collection

    if not TRAIN_DATA_PATH.exists():
        logger.warning(f"No training data at {TRAIN_DATA_PATH} — precedent index will be empty")
        return collection

    ids, documents, metadatas = [], [], []

    with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row      = json.loads(line)
                messages = row.get("messages", [])
                asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
                if not asst_msg:
                    continue

                label     = json.loads(asst_msg["content"])
                citation  = label.get("citation", "")
                case_name = label.get("case_name", "")
                holding   = label.get("holding", "")
                year      = label.get("year", 0)
                court     = label.get("court", "")
                outcome   = label.get("outcome", "")

                if not citation or not case_name or not holding:
                    continue

                doc_id = citation.replace(" ", "_").replace("/", "_")[:100]
                if doc_id in ids:
                    continue

                ids.append(doc_id)
                documents.append(f"{case_name}\n{holding}")
                metadatas.append({
                    "citation":  citation,
                    "case_name": case_name,
                    "year":      year or 0,
                    "court":     court,
                    "outcome":   outcome,
                })

            except (json.JSONDecodeError, KeyError):
                continue

    if not ids:
        logger.warning("No valid precedents found in training data")
        return collection

    # Batch upsert
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    logger.success(f"Precedent index: {len(ids)} cases indexed")
    return collection


# ── Step 5: Verify both indexes ───────────────────────────────────────────────

def verify_indexes(statute_col, precedent_col):
    logger.info("\nVerifying indexes...")

    # Test statute lookup
    results = statute_col.query(
        query_texts=["Indian Penal Code Section 302 murder punishment"],
        n_results=3,
    )
    logger.info("Test query: IPC Section 302")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        logger.info(f"  [{dist:.3f}] {meta.get('act', '')} — {doc[:80]}...")

    # Test precedent lookup
    if precedent_col.count() > 0:
        results2 = precedent_col.query(
            query_texts=["circumstantial evidence murder conviction"],
            n_results=3,
        )
        logger.info("\nTest query: precedent search")
        for doc, meta, dist in zip(
            results2["documents"][0],
            results2["metadatas"][0],
            results2["distances"][0],
        ):
            logger.info(f"  [{dist:.3f}] {meta.get('citation', '')} — "
                       f"{meta.get('case_name', '')[:50]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Nyaya-7B — Downloading Statutes & Building Knowledge Bases")
    logger.info("=" * 60)

    # ── Download statute texts ────────────────────────────────────────────────
    ipc_text          = download_ipc()
    constitution_text = download_constitution()
    other_acts        = download_other_acts()

    saved = save_statutes(ipc_text, constitution_text, other_acts)

    if not saved:
        logger.error(
            "No statute files saved — all HuggingFace downloads failed.\n"
            "This is unusual. Check your internet connection and try again."
        )
        return

    # ── Build ChromaDB indexes ────────────────────────────────────────────────
    logger.info("\nLoading embedding model...")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    logger.info("\nBuilding statute index...")
    statute_col = build_statute_index(ef)

    logger.info("\nBuilding precedent index...")
    precedent_col = build_precedent_index(ef)

    # ── Verify ────────────────────────────────────────────────────────────────
    if statute_col.count() > 0:
        verify_indexes(statute_col, precedent_col)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.success("Knowledge bases ready:")
    logger.success(f"  Statute index:   {statute_col.count()} chunks  → {CHROMA_STATUTE_PATH}")
    logger.success(f"  Precedent index: {precedent_col.count()} cases  → {CHROMA_PRECEDENT_PATH}")
    logger.info("\nNext step: python agents/graph.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()