import io
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict

import pdfplumber
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings

logger = logging.getLogger(__name__)


def _extract_text(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)


def _chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    while start < len(text):
        chunks.append(text[start: start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def ingest_document(file_bytes: bytes, filename: str) -> Dict:
    from qdrant_client import QdrantClient
    from app.services.embedder import embedder

    client = QdrantClient(url=settings.QDRANT_URL)

    logger.info(f"Ingesting document: {filename}")
    text = _extract_text(file_bytes)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {filename}")

    chunks = _chunk_text(text)
    logger.info(f"Created {len(chunks)} chunks from {filename}")

    embeddings = embedder.encode_batch(chunks)

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in existing:
        dim = len(embeddings[0])
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {settings.COLLECTION_NAME}")

    # Delete existing chunks for this file (re-upload support)
    try:
        client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
        )
    except Exception:
        pass  # Collection might be empty

    now = datetime.now(timezone.utc).isoformat()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text": chunks[i],
                "filename": filename,
                "chunk_index": i,
                "ingested_at": now,
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    logger.info(f"Uploaded {len(points)} chunks for {filename}")

    return {"filename": filename, "chunk_count": len(chunks), "char_count": len(text)}


def list_documents() -> List[Dict]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL)

    try:
        collections = [c.name for c in client.get_collections().collections]
        if settings.COLLECTION_NAME not in collections:
            return []

        docs: Dict[str, Dict] = {}
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                fname = record.payload.get("filename", "unknown")
                if fname not in docs:
                    docs[fname] = {
                        "filename": fname,
                        "chunk_count": 0,
                        "ingested_at": record.payload.get("ingested_at", ""),
                    }
                docs[fname]["chunk_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return list(docs.values())
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []


def delete_document(filename: str, collection: str = None):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL)
    col = collection or settings.COLLECTION_NAME
    client.delete(
        collection_name=col,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
    )
    logger.info(f"Deleted document: {filename} from {col}")


def list_documents_for_collection(collection: str) -> List[Dict]:
    """List documents in a specific collection (used for per-API-key isolation)."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL)
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection not in existing:
            return []

        docs: Dict[str, Dict] = {}
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                fname = record.payload.get("filename", "unknown")
                if fname not in docs:
                    docs[fname] = {
                        "filename": fname,
                        "chunk_count": 0,
                        "ingested_at": record.payload.get("ingested_at", ""),
                    }
                docs[fname]["chunk_count"] += 1
            if next_offset is None:
                break
            offset = next_offset
        return list(docs.values())
    except Exception as e:
        logger.error(f"Error listing documents for collection {collection}: {e}")
        return []


def ingest_document_to_collection(file_bytes: bytes, filename: str, collection: str) -> Dict:
    """Ingest into a specific collection (per-API-key isolation)."""
    original = settings.COLLECTION_NAME
    settings.COLLECTION_NAME = collection
    try:
        return ingest_document(file_bytes, filename)
    finally:
        settings.COLLECTION_NAME = original
