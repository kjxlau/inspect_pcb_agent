# src/data/qdrant_store.py
import logging
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "qdrant_db"
COLLECTION_NAME = "pcb_standards_precedents"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Cached instances
_client = None
_embedder = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(DB_PATH))
    return _client


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embedder


def query_defect_precedents(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Runs a semantic vector search over IPC standards and defect documents."""
    client = get_qdrant_client()
    embedder = get_embedder()

    if not client.collection_exists(COLLECTION_NAME):
        logger.warning(f"Collection {COLLECTION_NAME} does not exist.")
        return []

    query_vector = list(embedder.embed([query_text]))[0].tolist()

    # Use query_points (modern qdrant-client API)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    results = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append({
            "score": round(float(hit.score), 4),
            "text": payload.get("text", ""),
            "source": payload.get("source_file", "unknown"),
            "page": payload.get("page", None)
        })

    return results
