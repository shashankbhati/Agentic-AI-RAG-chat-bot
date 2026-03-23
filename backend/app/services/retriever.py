import logging
from typing import List, Dict, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class RetrieverService:
    """Handles all Qdrant vector search operations."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=settings.QDRANT_URL)
        return self._client

    def search(self, query: str, top_k: Optional[int] = None, collection: Optional[str] = None) -> List[Dict]:
        from app.services.embedder import embedder

        k = top_k or settings.TOP_K
        col = collection or settings.COLLECTION_NAME
        query_emb = embedder.encode(query)

        results = self.client.query_points(
            collection_name=col,
            query=query_emb,
            limit=k,
        )

        return [
            {
                "score": point.score,
                "text": point.payload.get("text", ""),
                "filename": point.payload.get("filename", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            }
            for point in results.points
        ]

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def get_collection_info(self) -> Dict:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if settings.COLLECTION_NAME in collections:
                info = self.client.get_collection(settings.COLLECTION_NAME)
                return {
                    "exists": True,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                }
            return {"exists": False, "vectors_count": 0, "points_count": 0}
        except Exception as e:
            return {"exists": False, "error": str(e), "vectors_count": 0, "points_count": 0}

    def delete_document_chunks(self, filename: str):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
        )
        logger.info(f"Deleted chunks for document: {filename}")


retriever = RetrieverService()
