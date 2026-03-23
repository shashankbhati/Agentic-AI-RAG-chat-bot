import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from app.config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict] = []
        self.created_at = _now()
        self.last_accessed = _now()

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": _now().isoformat(),
        })
        self.last_accessed = _now()

    def get_history(self) -> List[Dict]:
        return self.history

    def is_expired(self) -> bool:
        ttl = timedelta(hours=settings.SESSION_TTL_HOURS)
        return _now() - self.last_accessed > ttl


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            del self._sessions[session_id]
            return None
        return session

    def get_or_create(self, session_id: Optional[str]) -> tuple:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session_id, session
        new_id = self.create_session()
        return new_id, self._sessions[new_id]

    def cleanup_expired(self):
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


session_store = SessionStore()
