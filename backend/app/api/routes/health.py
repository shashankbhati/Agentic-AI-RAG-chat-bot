from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.retriever import retriever
from app.services.llm_service import llm_service
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check():
    qdrant_ok = retriever.health_check()
    collection_info = retriever.get_collection_info()

    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant="connected" if qdrant_ok else "disconnected",
        llm_provider=settings.LLM_PROVIDER,
        embed_model=settings.EMBED_MODEL_NAME,
        version="1.0.0",
        collection_info=collection_info,
    )
