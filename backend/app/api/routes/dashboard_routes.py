import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.session import get_db
from app.database.models import APIKey, UsageLog
from app.api.dependencies import require_api_key
from app.services.ingest_service import list_documents_for_collection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/dashboard/stats", summary="Usage stats for your API key")
def get_stats(
    api_key: APIKey = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    totals = (
        db.query(UsageLog.event_type, func.count(UsageLog.id).label("count"))
        .filter(UsageLog.api_key_id == api_key.id)
        .group_by(UsageLog.event_type)
        .all()
    )
    counts = {row.event_type: row.count for row in totals}

    recent = (
        db.query(UsageLog)
        .filter(UsageLog.api_key_id == api_key.id)
        .order_by(UsageLog.timestamp.desc())
        .limit(20)
        .all()
    )

    return {
        "api_key_name": api_key.name,
        "collection_name": api_key.collection_name,
        "totals": {
            "chat": counts.get("chat", 0),
            "search": counts.get("search", 0),
            "upload": counts.get("upload", 0),
            "delete": counts.get("delete", 0),
        },
        "recent_activity": [
            {
                "event_type": log.event_type,
                "document_name": log.document_name,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            }
            for log in recent
        ],
    }


@router.get("/dashboard/documents", summary="List documents for your API key's collection")
def get_documents(
    api_key: APIKey = Depends(require_api_key),
):
    docs = list_documents_for_collection(api_key.collection_name)
    return {"documents": docs, "total": len(docs), "collection": api_key.collection_name}
