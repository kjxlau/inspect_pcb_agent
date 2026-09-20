"""Ingestion & Indexing pipeline for IPC standards and defect literature into Qdrant."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
STANDARDS_DIR = BASE_DIR / "ipc_standards"
DB_PATH = BASE_DIR / "qdrant_db"
COLLECTION_NAME = "pcb_standards_precedents"

# Model setup (FastEmbed generates 384-d vectors locally without external API costs)
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


def extract_chunks_from_pdf(pdf_path: Path, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Extracts text from a PDF and produces overlapping text chunks with page tracking."""
    logger.info(f"Reading PDF: {pdf_path.name}")
    chunks: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(str(pdf_path))
        full_text = ""
        page_markers = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            start_char = len(full_text)
            full_text += text + "\n"
            page_markers.append((start_char, len(full_text), page_idx + 1))

        # Sliding window chunking
        stride = chunk_size - overlap
        for i in range(0, len(full_text), stride):
            chunk_text = full_text[i:i + chunk_size].strip()
            if len(chunk_text) < 40:  # Skip trivial fragments
                continue

            # Identify source page
            page_num = 1
            for start, end, p_num in page_markers:
                if start <= i <= end:
                    page_num = p_num
                    break

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file": pdf_path.name,
                    "document_type": "standard_spec_or_guide",
                    "page": page_num
                }
            })
    except Exception as exc:
        logger.error(f"Error parsing PDF {pdf_path.name}: {exc}")

    return chunks


def extract_chunks_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """Extracts structured defect criteria rules and standards from JSON specifications."""
    logger.info(f"Reading JSON: {json_path.name}")
    chunks: List[Dict[str, Any]] = []
    try:
        # Use utf-8-sig to automatically handle any Windows BOM
        with open(json_path, "r", encoding="utf-8-sig") as f:
            raw_content = f.read().strip()
            if not raw_content:
                logger.warning(f"File {json_path.name} is empty. Skipping.")
                return []
            data = json.loads(raw_content)

        # 1. Custom parser for "ipc_standards_framework" schema
        if isinstance(data, dict) and "ipc_standards_framework" in data:
            framework = data["ipc_standards_framework"]
            context_desc = framework.get("context", "Surface Mount Assembly & Inspection")
            standards_list = framework.get("standards", [])

            for item in standards_list:
                std_id = item.get("standard_id", "Unknown Standard")
                title = item.get("title", "")
                category = item.get("category", "")
                role = item.get("role", "")
                relevance = item.get("relevance", "")
                refs = "\n- ".join(item.get("key_references", []))

                chunk_text = (
                    f"Standard: {std_id} - {title}\n"
                    f"Framework Context: {context_desc}\n"
                    f"Category: {category}\n"
                    f"Role: {role}\n"
                    f"Relevance & Inspection Criteria: {relevance}\n"
                    f"Key Section References:\n- {refs}"
                )

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_file": json_path.name,
                        "document_type": "ipc_standards_spec",
                        "standard_id": std_id,
                        "category": category
                    }
                })

        # 2. Fallback parser for generic JSON lists
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                chunks.append({
                    "text": json.dumps(item, indent=2),
                    "metadata": {
                        "source_file": json_path.name,
                        "document_type": "json_rule",
                        "rule_index": idx
                    }
                })

        # 3. Fallback parser for generic key-value JSON dicts
        elif isinstance(data, dict):
            for k, v in data.items():
                chunks.append({
                    "text": f"Standard Specification [{k}]:\n{json.dumps(v, indent=2)}",
                    "metadata": {
                        "source_file": json_path.name,
                        "document_type": "json_rule",
                        "rule_key": str(k)
                    }
                })

    except Exception as exc:
        logger.error(f"Error reading JSON {json_path.name}: {exc}")

    return chunks


def populate():
    """Builds and indexes vector database using FastEmbed and persistent local Qdrant."""
    from fastembed import TextEmbedding

    # 1. Collect all documents
    all_chunks: List[Dict[str, Any]] = []

    for file_path in STANDARDS_DIR.glob("*"):
        if file_path.suffix.lower() == ".pdf":
            all_chunks.extend(extract_chunks_from_pdf(file_path))
        elif file_path.suffix.lower() == ".json":
            all_chunks.extend(extract_chunks_from_json(file_path))

    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    if not all_chunks:
        logger.warning("No documents found to index in src/data/ipc_standards/")
        return

    # 2. Embed texts
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = list(embedder.embed(texts))

    # 3. Initialize persistent local Qdrant
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(DB_PATH))

    try:
        # Recreate collection
        if client.collection_exists(COLLECTION_NAME):
            logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        # 4. Upload in batches
        batch_size = 64
        points = []
        for idx, (chunk, vector) in enumerate(zip(all_chunks, embeddings)):
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
            )

            if len(points) >= batch_size:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

        logger.info(
            f"Successfully indexed {len(all_chunks)} chunks into Qdrant "
            f"collection '{COLLECTION_NAME}' at {DB_PATH}"
        )
    finally:
        # Release SQLite/RocksDB lock on disk
        client.close()


if __name__ == "__main__":
    populate()
