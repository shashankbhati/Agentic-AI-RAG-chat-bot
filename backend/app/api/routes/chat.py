import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.models.schemas import ChatRequest, ChatResponse, SessionHistoryResponse, ChatMessage
from app.services.retriever import RetrieverService
from app.services.llm_service import llm_service
from app.services.session_store import session_store
from app.services.usage_tracker import track
from app.api.dependencies import optional_api_key
from app.database.session import get_db
from app.database.models import APIKey
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _collection(api_key: Optional[APIKey]) -> str:
    return api_key.collection_name if api_key else settings.COLLECTION_NAME


@router.post("/chat", response_model=ChatResponse, summary="Chat with RAG (non-streaming)")
def chat(
    request: ChatRequest,
    api_key: Optional[APIKey] = Depends(optional_api_key),
    db: Session = Depends(get_db),
):
    try:
        session_id, session = session_store.get_or_create(request.session_id)
        retriever = RetrieverService()

        results = retriever.search(request.query, collection=_collection(api_key))
        contexts = [r["text"] for r in results]
        sources = sorted(set(r["filename"] for r in results))
        context_text = "\n\n---\n\n".join(contexts)

        answer = llm_service.generate(request.query, context_text, session.get_history())

        session.add_message("user", request.query)
        session.add_message("assistant", answer)

        if api_key:
            track(db, api_key.id, "chat")

        return ChatResponse(
            session_id=session_id,
            query=request.query,
            answer=answer,
            retrieved_chunks=contexts,
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", summary="Chat with RAG (streaming via SSE)")
def chat_stream(
    request: ChatRequest,
    api_key: Optional[APIKey] = Depends(optional_api_key),
    db: Session = Depends(get_db),
):
    try:
        session_id, session = session_store.get_or_create(request.session_id)
        retriever = RetrieverService()

        results = retriever.search(request.query, collection=_collection(api_key))
        contexts = [r["text"] for r in results]
        sources = sorted(set(r["filename"] for r in results))
        context_text = "\n\n---\n\n".join(contexts)
        history = session.get_history()

        if api_key:
            track(db, api_key.id, "chat")

        def event_stream():
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'sources': sources})}\n\n"
            full_answer = []
            try:
                for chunk in llm_service.stream(request.query, context_text, history):
                    full_answer.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
                return
            session.add_message("user", request.query)
            session.add_message("assistant", "".join(full_answer))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream setup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{session_id}/history", response_model=SessionHistoryResponse)
def get_history(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return SessionHistoryResponse(
        session_id=session_id,
        history=[ChatMessage(**m) for m in session.get_history()],
    )


@router.delete("/chat/{session_id}")
def clear_session(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    session.history.clear()
    return {"message": "Session cleared"}
