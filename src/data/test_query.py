"""Verification and query test script for IPC Standards & Defect Literature in Qdrant."""

from __future__ import annotations

import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_store import query_defect_precedents, DB_PATH, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def inspect_database():
    """Inspects the local Qdrant collection status and prints sample records."""
    print("=" * 70)
    print(" 1. INSPECTING LOCAL QDRANT STANDARDS COLLECTION")
    print("=" * 70)
    print(f"Database directory: {DB_PATH.resolve()}")
    print(f"Target collection:  {COLLECTION_NAME}")

    client = QdrantClient(path=str(DB_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        print(f"\n[ERROR] Collection '{COLLECTION_NAME}' does not exist!")
        print("Please run `python src/data/populate_qdrant.py` first to index your documents.")
        return False

    collection_info = client.get_collection(COLLECTION_NAME)
    points_count = collection_info.points_count
    print(f"Total indexed chunks/points: {points_count}")

    if points_count == 0:
        print("\n[WARNING] Collection exists but contains 0 points. Run populate_qdrant.py.")
        return False

    # Fetch 2 sample records
    sample_records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2,
        with_payload=True
    )

    print("\n--- Sample Indexed Records ---")
    for record in sample_records:
        payload = record.payload or {}
        source = payload.get("source_file", "unknown")
        doc_type = payload.get("document_type", "unknown")
        page = payload.get("page", "N/A")
        preview = payload.get("text", "").replace("\n", " ")[:150]
        print(f"\n[Point ID {record.id}]")
        print(f"  Source:   {source} (Type: {doc_type}, Page: {page})")
        print(f"  Excerpt:  {preview}...")

    client.close()
    return True


def test_semantic_search():
    """Runs realistic engineering inspection queries against the indexed knowledge base."""
    print("\n" + "=" * 70)
    print(" 2. TESTING SEMANTIC SEARCH QUERIES (IPC-A-610 & DEFECT RAG)")
    print("=" * 70)

    test_queries = [
        "What causes tombstoning on discrete chip components and how to prevent it?",
        "IPC-A-610 Class 2 maximum allowable side overhang tolerance percentage",
        "Solder starvation insufficient wetting minimum fillet height requirement",
        "Automated Optical Inspection 3D laser height detection for missing part"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n----------------------------------------------------------------------")
        print(f"[Query {idx}] \"{query}\"")
        print(f"----------------------------------------------------------------------")
        
        results = query_defect_precedents(query_text=query, top_k=2)

        if not results:
            print("  No matching documents found.")
            continue

        for rank, hit in enumerate(results, 1):
            score = hit.get("score", 0.0)
            source = hit.get("source", "Unknown Document")
            page = f" (Page {hit['page']})" if hit.get("page") else ""
            excerpt = hit.get("text", "").strip()
            
            # Print condensed snippet
            condensed = "\n    ".join(excerpt.splitlines()[:5])
            print(f"  Result #{rank} | Similarity: {score:.4f} | Source: {source}{page}")
            print(f"    {condensed}")
            if len(excerpt.splitlines()) > 5:
                print("    ...")


if __name__ == "__main__":
    is_valid = inspect_database()
    if is_valid:
        test_semantic_search()
        print("\n" + "=" * 70)
        print(" VERIFICATION COMPLETE: Standards & literature are query-ready.")
        print("=" * 70 + "\n")