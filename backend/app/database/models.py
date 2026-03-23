import secrets
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database.session import Base


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)

    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String, unique=True, index=True, nullable=False, default=lambda: f"rag-{secrets.token_urlsafe(32)}")
    name = Column(String, default="Default")
    collection_name = Column(String, nullable=False, default=lambda: f"rag_{secrets.token_hex(8)}")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key_rel", cascade="all, delete-orphan")


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    event_type = Column(String, nullable=False)   # "chat", "search", "upload", "delete"
    document_name = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=_now)

    api_key_rel = relationship("APIKey", back_populates="usage_logs")
