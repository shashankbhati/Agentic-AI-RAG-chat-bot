import logging
from sqlalchemy.orm import Session
from app.database.models import UsageLog

logger = logging.getLogger(__name__)


def track(db: Session, api_key_id: int, event_type: str, document_name: str = None):
    try:
        log = UsageLog(
            api_key_id=api_key_id,
            event_type=event_type,
            document_name=document_name,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.warning(f"Usage tracking failed: {e}")
        db.rollback()
