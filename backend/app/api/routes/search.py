import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.retriever import RetrieverService
from app.services.usage_tracker import track
from app.api.dependencies import optional_api_key
from app.database.session import get_db
from app.database.models import APIKey
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse, summary="Semantic search over documents")
def search(
    request: SearchRequest,
    api_key: Optional[APIKey] = Depends(optional_api_key),
    db: Session = Depends(get_db),
):
    try:
        collection = api_key.collection_name if api_key else settings.COLLECTION_NAME
        retriever = RetrieverService()
        raw = retriever.search(request.query, top_k=request.top_k, collection=collection)

        if api_key:
            track(db, api_key.id, "search")

        return SearchResponse(query=request.query, results=[SearchResult(**r) for r in raw])
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed. Check that Qdrant is running.")
